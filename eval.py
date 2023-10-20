# Testing
import os
# todo 根据实际情况修改
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import argparse
from thop import profile
from net.net import net
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *

# Testing settings
parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
# 测试集位置
# parser.add_argument('--data_test', type=str, default='dataset/test/LOL-test/raw')
parser.add_argument('--data_test', type=str, default='dataset/test/SICE-test/image')
# parser.add_argument('--data_test', type=str, default='dataset/test/MEF')
# 预训练模型位置
# parser.add_argument('--model', default='weights/PairLIE.pth', help='Pretrained base model')
parser.add_argument('--model', default='weights/epoch_400.pth', help='Pretrained base model')
# 测试输出位置
# parser.add_argument('--output_folder', type=str, default='results/MEF/')
# parser.add_argument('--output_folder', type=str, default='results/LOL/')
parser.add_argument('--output_folder', type=str, default='results/SICE/')
opt = parser.parse_args()

# 加载测试集
print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
# 加载预训练模型
print('===> Building model')
model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

def eval():
    torch.set_grad_enabled(False)# 禁用梯度计算，减少计算开销
    model.eval()
    print('\nEvaluation:')
    for batch in testing_data_loader:
        with torch.no_grad():# 在没有梯度计算的上下文中运行以下代码块
            input, name = batch[0], batch[1]# 从当前批次中提取输入数据和数据名称
        input = input.cuda()
        print(name)# 打印数据名称

        with torch.no_grad():
            L, R, X = model(input)
            D = input- X # difference map
            # 进行图像增强
            I = torch.pow(L,0.2) * R  # g(L)--default=0.2, LOL=0.14.
            # flops, params = profile(model, (input,))
            # print('flops: ', flops, 'params: ', params)

        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)
            os.mkdir(opt.output_folder + 'L/')
            os.mkdir(opt.output_folder + 'R/')
            os.mkdir(opt.output_folder + 'I/')  
            os.mkdir(opt.output_folder + 'D/')

        # 将计算结果从GPU移动回CPU，因为接下来要使用PIL库保存图像，PIL通常需要在CPU上操作
        L = L.cpu()
        R = R.cpu()
        I = I.cpu()
        D = D.cpu()        
        # 将图像转换为PIL图像对象
        L_img = transforms.ToPILImage()(L.squeeze(0))
        R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))                
        D_img = transforms.ToPILImage()(D.squeeze(0))  

        L_img.save(opt.output_folder + '/L/' + name[0])
        R_img.save(opt.output_folder + '/R/' + name[0])
        I_img.save(opt.output_folder + '/I/' + name[0])  
        D_img.save(opt.output_folder + '/D/' + name[0])                       
    # 重新启用梯度计算，以便在评估函数之后可以继续进行训练等操作
    torch.set_grad_enabled(True)
# 进行测试
eval()


