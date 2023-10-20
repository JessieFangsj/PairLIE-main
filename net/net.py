import torch
import torch.nn as nn

# illumination
class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        # 使用了 nn.Sequential 封装一系列层
        self.L_net = nn.Sequential(
            # 对输入进行边界填充，参数为1
            nn.ReflectionPad2d(1),# 扩展图像的大小，以适应卷积等操作、保持图像边界信息
            # input_channel=3,输出通道数=num,卷积核大小=3x3，步幅=1，填充=0
            nn.Conv2d(3, num, 3, 1, 0),# 处理特征图
            # ReLU激活函数，输出范围[0,正无穷)，线性和非线性结合的激活函数，导致稀疏激活性
            nn.ReLU(),# 引入非线性，以便网络可以学习复杂的特征
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            # 输输出通道数为 1，用于生成一通道的图像(RGB的illumination分量相同)
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        # 使用sigmoid函数对模型的输出进行激活，将其限制在(0,1)范围内，非线性激活函数，不会导致稀疏激活性
        return torch.sigmoid(self.L_net(input))

# reflectance
class R_net(nn.Module):
    def __init__(self, num=64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),# 输出三通道
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input))
    
# preprocessed
class N_net(nn.Module):
    def __init__(self, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            #在输入图像的边缘周围添加一圈像素，以处理卷积操作可能导致的边缘信息丢失
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.N_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        self.L_net = L_net(num=64)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)        

    def forward(self, input):
        x = self.N_net(input)# preprocessed
        L = self.L_net(x)# illumination
        R = self.R_net(x)# refletance
        return L, R, x
