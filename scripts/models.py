import torch
import torch.nn as nn 


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, h=128,w=128):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        
        self.resample = resample
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
         
        if resample == 'down':
            
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(input_dim)
            self.bn3 = nn.BatchNorm2d(input_dim)
            
            self.conv_shortcut = nn.Conv2d(input_dim, output_dim, kernel_size = 3,stride = 1,padding = 1)
            self.up_down_pool_1 = nn.AvgPool2d(2,stride = 2)
            self.conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size = kernel_size,stride = 1,padding = 1)
            self.up_down_pool_2 = nn.AvgPool2d(2,stride = 2)
            self.conv_2 = nn.Conv2d(input_dim, input_dim, kernel_size = kernel_size,stride = 1,padding = 1)
            self.conv_3 = nn.Conv2d(input_dim, output_dim, kernel_size = kernel_size,stride = 1,padding = 1)
        
        elif resample == 'up':
            
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
            self.bn3 = nn.BatchNorm2d(output_dim)
            
            self.conv_shortcut = nn.Conv2d(output_dim, output_dim, kernel_size = 3,stride = 1,padding = 1)
            self.up_down_pool_1 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size = 2,stride = 2)
            self.conv_1 = nn.Conv2d(input_dim, output_dim, kernel_size = kernel_size,stride = 1,padding = 1)

            self.up_down_pool_2 = nn.ConvTranspose2d(output_dim, output_dim, kernel_size = 2,stride = 2)
            self.conv_2 = nn.Conv2d(output_dim, output_dim, kernel_size = kernel_size,stride = 1,padding = 1)
            self.conv_3 = nn.Conv2d(output_dim, output_dim, kernel_size = kernel_size,stride = 1,padding = 1)
        
        else:
            raise Exception('invalid resample value')

            
    def forward(self, input):
        shortcut = self.conv_shortcut(self.up_down_pool_1(input))
        output = self.bn1(input)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.up_down_pool_2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.conv_3(output)
        
        return shortcut + output

class Encoder_Decoder_128(nn.Module):
    def __init__(self, dim=32):
        super(Encoder_Decoder_128, self).__init__()
        
        self.h,self.w = 128,128
        self.dim = dim
        self.conv1_enc = nn.Conv2d(3, self.dim, kernel_size = 3,stride = 1,padding = 1)
        self.rb1_enc = ResidualBlock(self.dim, 1*self.dim, 3, resample = 'down',h=self.h,w=self.w)
        self.h,self.w = self.change_hw(self.h,self.w)
        self.rb2_enc = ResidualBlock(1*self.dim, 2*self.dim, 3, resample = 'down', h=self.h,w=self.w)
        self.h,self.w = self.change_hw(self.h,self.w)
        self.rb3_enc = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', h=self.h,w=self.w)
        self.h,self.w = self.change_hw(self.h,self.w)
        self.rb4_enc = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', h=self.h,w=self.w)
        self.h,self.w = self.change_hw(self.h,self.w)
        self.rb5_enc = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', h=self.h,w=self.w)
        self.h,self.w = self.change_hw(self.h,self.w,last = True)
        
        self.rb1_dec = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2_dec = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3_dec = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4_dec = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.rb5_dec = ResidualBlock(1*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)
        
        self.conv_dec = nn.Conv2d(self.dim, 3, kernel_size = 3,stride = 1,padding = 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        
        output = input.contiguous()
        output = self.conv1_enc(output)
        output = self.rb1_enc(output)
        output = self.rb2_enc(output)
        output = self.rb3_enc(output)
        output = self.rb4_enc(output)
        output = self.rb5_enc(output)
        
        output = self.rb1_dec(output)
        output = self.rb2_dec(output)
        output = self.rb3_dec(output)
        output = self.rb4_dec(output)
        output = self.rb5_dec(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv_dec(output)
        output = self.tanh(output)
        return output

    
    def change_hw(self,h,w,last = False):
        if((h/2)% 2 != 0 and not(last)):
            h = (h//2)-1
        else:
            h = h//2
        
        if((w/2)% 2 != 0 and not(last)):
            w = (w//2)-1
        else:
            w = w//2
        
        return h,w
