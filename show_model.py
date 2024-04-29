# from torchsummary import summary
from networks.ResNet_3D_CPM import Resnet18
# from torchviz import make_dot   
# import torch 
# from torch import nn 
# model = Resnet18().cuda().eval()
# # summary(model.eval(), (1, 96, 96, 96), device='cuda')
# # print(model)
# sample_input = torch.randn(1, 1, 96, 96, 96).cuda()  # 输入张量的形状应该与您的模型的输入张量形状相匹配

# # 获取模型的输出
# output = model(sample_input)

# # 使用torchviz的make_dot函数可视化模型
# dot = make_dot(output, params=dict(model.named_parameters()))

# # 保存可视化结果为PDF或任何其他格式
# dot.render("cpm_model", format="pdf")

# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import netron
import torch.onnx

model = Resnet18(se=False).cuda().eval()
sample_input = torch.randn(1, 1, 96, 96, 96).cuda() 
# summary(model.eval(), (1, 96, 96, 96), device='cuda')
# myNet = resnet18()  # 实例化 resnet18
# x = torch.randn(16, 3, 40, 40)  # 随机生成一个输入
modelData = "./demo.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
torch.onnx.export(model, sample_input, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(modelData)  # 输出网络结构
