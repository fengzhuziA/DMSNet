# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from torch.nn import Softmax ,Dropout, LayerNorm
from pdb import set_trace as stx
import copy
from einops import rearrange, repeat
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class FFblock(nn.Module):
    def __init__(self,in_feat,bias=True):
        super(FFblock, self).__init__()
        c=in_feat
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.conv2 = nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, stride=1, groups=2*c,
                               bias=True)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
      
    def forward(self,x):
        input=x
        x = self.norm1(x)
        x= self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x= self.conv3(x)+input

        return x
class RNN_SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RNN_SAM, self).__init__()
     
        self.conv1 = FFblock(n_feat)
        self.conv1_1 = FFblock(n_feat)
        self.convM_1 = FFblock(n_feat)
        self.convF = FFblock(n_feat)
        
        self.scaM = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.scaF = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv3_1 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img,hidden_state=None):
        if hidden_state is None:
            start_input = x
        else:
            start_input = hidden_state+x
            
       
        img = self.conv2(start_input) + x_img
        
        F_scores =self.scaF(self.conv3_1(img))
        F_feature = self.conv1_1(start_input)
        F_feature = F_feature*F_scores + start_input
        
        M_scores =self.conv3(img)+F_feature
        M_scores =self.scaM(self.convM_1(M_scores))
        M_feature = self.conv1(start_input)
        M_feature = M_feature*M_scores + F_feature
    
       
        return F_feature, img,M_feature


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Memory_util(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)


    def forward(self, q,k,v):
        x = q+k

        x = self.norm1(x)
        v = self.norm2(v)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = v * self.sca(x)
 

        return x

class Multi_hop_query(nn.Module):
    def __init__(self,in_feat, s_window=8, kernel_size=3,hops=1, bias=False):
        super(Multi_hop_query, self).__init__()
        self.memory_untils = nn.ModuleList()
        self.query_convs = nn.ModuleList()
        for i in range(hops):
            memory_util = Memory_util(in_feat)
            self.memory_untils.append(memory_util)
          
            query_conv = FFblock(in_feat)
            self.query_convs.append(query_conv)
    def forward(self,q,k,v):
        for hop in range(len(self.memory_untils)):
            query_out = self.memory_untils[hop](q,k,v)
           
            q=query_out+self.query_convs[hop](q)
        return q
class Encoder(nn.Module):
    
    def __init__(self,  width=16,enc_blk_nums=[]):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        ##HNM blocks
        self.memory1 = Multi_hop_query(width,s_window=16)
        self.memory2 = Multi_hop_query(width*2,s_window=8)
        self.memory3 = Multi_hop_query(width*4,s_window=4)
        self.memory4 = Multi_hop_query(width*8,s_window=2)
    def forward(self, x, encoder_outs=None, decoder_outs=None):
        
        enc1 = self.encoders[0](x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 +self.memory1(enc1,encoder_outs[0],decoder_outs[0])
            
        x = self.downs[0](enc1)

        enc2 = self.encoders[1](x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.memory2(enc2,encoder_outs[1],decoder_outs[1])

        x = self.downs[1](enc2)

        enc3 = self.encoders[2](x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.memory3(enc3,encoder_outs[2],decoder_outs[2])
        x = self.downs[2](enc3)
        
        enc4 = self.encoders[3](x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc4 = enc4 + self.memory4(enc4,encoder_outs[3],decoder_outs[3])
            
        neck_out = self.downs[3](enc4)

        return [enc1, enc2, enc3,enc4],neck_out  
class Middle_block(nn.Module):   
    def __init__(self,  width=16,middle_blk_nums=1):
        super().__init__()
        self.middle_blks = nn.ModuleList()
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(width*16) for _ in range(middle_blk_nums)]
            )
    def forward(self, x):
        x = self.middle_blks(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self,  width=16,dec_blk_nums=[]):
        super().__init__()   
        chan = width*16
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
    def forward(self,middle_out,encs):
        
        enc1,enc2,enc3,enc4=encs
        
        x = self.ups[0](middle_out) +enc4
        dec4 = self.decoders[0](x)
        
        x = self.ups[1](dec4) + enc3
        dec3 = self.decoders[1](x)
        
        x = self.ups[2](dec3) + enc2
        dec2 = self.decoders[2](x)
        
        x = self.ups[3](dec2) + enc1
        dec1 = self.decoders[3](x)
        return [dec1,dec2,dec3,dec4]

        
        

class NAFNet(nn.Module):

    def __init__(self,time_step=3, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.time_step = time_step
        self.intros = nn.ModuleList()
        self.endings = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.sams = nn.ModuleList()
        
        for _ in range(self.time_step):
            intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
            encoder = Encoder(width=width,enc_blk_nums=enc_blk_nums)
            decoder = Decoder(width=width,dec_blk_nums=dec_blk_nums)
            middle_blk = Middle_block(width=width,middle_blk_nums=middle_blk_num)
            sam = RNN_SAM(n_feat=width,kernel_size=3,bias=True)
        
            
            self.intros.append(intro)
            self.encoders.append(encoder)
            self.middle_blks.append(middle_blk)
            self.decoders.append(decoder)
            self.sams.append(sam)
         
            
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        xs = []
        for _ in range(self.time_step):
            xs.append(copy.deepcopy(inp))

        encoder_outs = None
        decoder_outs = None
        hidden_satrt = None
        hidden_end = None
        imgs = []
     
        
        for i in range(self.time_step):
            shallow_input = self.intros[i](xs[i])
            if hidden_satrt is not None:
                shallow_input += hidden_satrt
            encoder_outs,middle_input = self.encoders[i](shallow_input,encoder_outs,decoder_outs)
            middle_out = self.middle_blks[i](middle_input)
            decoder_outs = self.decoders[i](middle_out,encoder_outs)
            hidden_satrt,img ,hidden_end= self.sams[i](decoder_outs[0],xs[i],hidden_end)
            imgs.append(img[:, :, :H, :W])
            
        return imgs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class DMSNetbaseLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    from thop import profile
    img_channel = 3
    width = 32
    time_step=3
   

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    device = torch.device('cuda')
    net = RNN_SAM(n_feat=64,kernel_size=3,bias=True)


    
    x = torch.randn(1,3, 256, 256)
    f = torch.randn(1,64, 256, 256)
    y=net(f,x)
    x=torch.randn(1,1,3,256,256)
    flops, params = profile(net, inputs=(f,x), verbose=False)
    print('[profile] computation cost: {:.2f} GMACs, parameters: {:.2f} M'.format(
            flops / 10 ** 9, params / 10 ** 6))
    