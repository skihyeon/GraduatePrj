import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

 
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        # self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//3 , kernel_size= 1)
        # self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//3 , kernel_size= 1)
        
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Attn_conv(nn.Module):
    def __init__(self):
        super(Attn_conv,self).__init__()
        p = 1
        self.pool = nn.MaxPool2d(2)
        self.ReLU = nn.ReLU()
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv1_1 = nn.Conv2d(3, 32, 3, padding = p)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding = p)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = p)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding = p)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding = p)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding = p)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding = p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding = p)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding = p)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = p)

        self.attn = Self_Attn(32, activation = nn.ReLU())

    def forward(self, input):
        x = self.bn1_1(self.ReLU(self.conv1_1(input)))
        x = self.bn1_2(self.ReLU(self.conv1_2(x)))
        x = self.pool(x)

        attn1,_ = self.attn(x) 

        attn2_1 = self.bn2_1(self.ReLU(self.conv2_1(attn1)))
        attn2_2 = self.bn2_2(self.ReLU(self.conv2_2(attn2_1)))
        attn2 = self.pool(attn2_2)

        attn3_1 = self.bn3_1(self.ReLU(self.conv3_1(attn2)))
        attn3_2 = self.bn3_2(self.ReLU(self.conv3_2(attn3_1)))
        attn3 = self.pool(attn3_2)

        attn4_1 = self.bn4_1(self.ReLU(self.conv4_1(attn3)))
        attn4_2 = self.bn4_2(self.ReLU(self.conv4_2(attn4_1)))
        attn4 = self.pool(attn4_2)

        attn5_1 = self.bn5_1(self.ReLU(self.conv5_1(attn4)))
        attn5_2 = self.bn5_2(self.ReLU(self.conv5_2(attn5_1)))
        # attn5 = self.pool(attn5_2)
        return attn5_2