import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        self.conv2d1 = Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        self.layernorm0 = LayerNorm((96,), eps=1e-06, elementwise_affine=True)
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)
        self.conv2d2 = Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        self.layernorm1 = LayerNorm((96,), eps=1e-06, elementwise_affine=True)
        self.linear2 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.linear3 = Linear(in_features=384, out_features=96, bias=True)
        self.conv2d3 = Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        self.layernorm2 = LayerNorm((96,), eps=1e-06, elementwise_affine=True)
        self.linear4 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.linear5 = Linear(in_features=384, out_features=96, bias=True)
        self.conv2d4 = Conv2d(96, 192, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d5 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        self.layernorm3 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        self.linear6 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.linear7 = Linear(in_features=768, out_features=192, bias=True)
        self.conv2d6 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        self.layernorm4 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        self.linear8 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.linear9 = Linear(in_features=768, out_features=192, bias=True)
        self.conv2d7 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        self.layernorm5 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        self.linear10 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.linear11 = Linear(in_features=768, out_features=192, bias=True)
        self.conv2d8 = Conv2d(192, 384, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d9 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm6 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear12 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.linear13 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d10 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm7 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d11 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm8 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear16 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.linear17 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d12 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm9 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d13 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm10 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear20 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.linear21 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d14 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm11 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear22 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.linear23 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d15 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm12 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear24 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.linear25 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d16 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm13 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear26 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.linear27 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d17 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        self.layernorm14 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear28 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.linear29 = Linear(in_features=1536, out_features=384, bias=True)
        self.conv2d18 = Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d19 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        self.layernorm15 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear30 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu15 = GELU(approximate='none')
        self.linear31 = Linear(in_features=3072, out_features=768, bias=True)
        self.conv2d20 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        self.layernorm16 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear32 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.linear33 = Linear(in_features=3072, out_features=768, bias=True)
        self.conv2d21 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        self.layernorm17 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.flatten0 = Flatten(start_dim=1, end_dim=-1)
        self.linear36 = Linear(in_features=768, out_features=1000, bias=True)
        self.weight0 = torch.rand(torch.Size([96])).to(torch.float32)
        self.bias0 = torch.rand(torch.Size([96])).to(torch.float32)
        self.layer_scale0 = torch.rand(torch.Size([96, 1, 1])).to(torch.float32)
        self.layer_scale1 = torch.rand(torch.Size([96, 1, 1])).to(torch.float32)
        self.layer_scale2 = torch.rand(torch.Size([96, 1, 1])).to(torch.float32)
        self.weight1 = torch.rand(torch.Size([96])).to(torch.float32)
        self.bias1 = torch.rand(torch.Size([96])).to(torch.float32)
        self.layer_scale3 = torch.rand(torch.Size([192, 1, 1])).to(torch.float32)
        self.layer_scale4 = torch.rand(torch.Size([192, 1, 1])).to(torch.float32)
        self.layer_scale5 = torch.rand(torch.Size([192, 1, 1])).to(torch.float32)
        self.weight2 = torch.rand(torch.Size([192])).to(torch.float32)
        self.bias2 = torch.rand(torch.Size([192])).to(torch.float32)
        self.layer_scale6 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale7 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale8 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale9 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale10 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale11 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale12 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale13 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.layer_scale14 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)
        self.weight3 = torch.rand(torch.Size([384])).to(torch.float32)
        self.bias3 = torch.rand(torch.Size([384])).to(torch.float32)
        self.layer_scale15 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)
        self.layer_scale16 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)
        self.layer_scale17 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)
        self.weight4 = torch.rand(torch.Size([768])).to(torch.float32)
        self.bias4 = torch.rand(torch.Size([768])).to(torch.float32)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=x1.permute(0, 2, 3, 1)
        x5=torch.nn.functional.layer_norm(x2, (96,),weight=self.weight0, bias=self.bias0, eps=1e-06)
        x6=x5.permute(0, 3, 1, 2)
        x8=self.conv2d1(x6)
        x9=torch.permute(x8, [0, 2, 3, 1])
        x10=self.layernorm0(x9)
        x11=self.linear0(x10)
        x12=self.gelu0(x11)
        x13=self.linear1(x12)
        x14=torch.permute(x13, [0, 3, 1, 2])
        x15=operator.mul(self.layer_scale0, x14)
        x16=stochastic_depth(x15, 0.0, 'row', False)
        x17=operator.add(x16, x6)
        x19=self.conv2d2(x17)
        x20=torch.permute(x19, [0, 2, 3, 1])
        x21=self.layernorm1(x20)
        x22=self.linear2(x21)
        x23=self.gelu1(x22)
        x24=self.linear3(x23)
        x25=torch.permute(x24, [0, 3, 1, 2])
        x26=operator.mul(self.layer_scale1, x25)
        x27=stochastic_depth(x26, 0.0058823529411764705, 'row', False)
        x28=operator.add(x27, x17)
        x30=self.conv2d3(x28)
        x31=torch.permute(x30, [0, 2, 3, 1])
        x32=self.layernorm2(x31)
        x33=self.linear4(x32)
        x34=self.gelu2(x33)
        x35=self.linear5(x34)
        x36=torch.permute(x35, [0, 3, 1, 2])
        x37=operator.mul(self.layer_scale2, x36)
        x38=stochastic_depth(x37, 0.011764705882352941, 'row', False)
        x39=operator.add(x38, x28)
        x40=x39.permute(0, 2, 3, 1)
        x43=torch.nn.functional.layer_norm(x40, (96,),weight=self.weight1, bias=self.bias1, eps=1e-06)
        x44=x43.permute(0, 3, 1, 2)
        x45=self.conv2d4(x44)
        x47=self.conv2d5(x45)
        x48=torch.permute(x47, [0, 2, 3, 1])
        x49=self.layernorm3(x48)
        x50=self.linear6(x49)
        x51=self.gelu3(x50)
        x52=self.linear7(x51)
        x53=torch.permute(x52, [0, 3, 1, 2])
        x54=operator.mul(self.layer_scale3, x53)
        x55=stochastic_depth(x54, 0.017647058823529415, 'row', False)
        x56=operator.add(x55, x45)
        x58=self.conv2d6(x56)
        x59=torch.permute(x58, [0, 2, 3, 1])
        x60=self.layernorm4(x59)
        x61=self.linear8(x60)
        x62=self.gelu4(x61)
        x63=self.linear9(x62)
        x64=torch.permute(x63, [0, 3, 1, 2])
        x65=operator.mul(self.layer_scale4, x64)
        x66=stochastic_depth(x65, 0.023529411764705882, 'row', False)
        x67=operator.add(x66, x56)
        x69=self.conv2d7(x67)
        x70=torch.permute(x69, [0, 2, 3, 1])
        x71=self.layernorm5(x70)
        x72=self.linear10(x71)
        x73=self.gelu5(x72)
        x74=self.linear11(x73)
        x75=torch.permute(x74, [0, 3, 1, 2])
        x76=operator.mul(self.layer_scale5, x75)
        x77=stochastic_depth(x76, 0.029411764705882353, 'row', False)
        x78=operator.add(x77, x67)
        x79=x78.permute(0, 2, 3, 1)
        x82=torch.nn.functional.layer_norm(x79, (192,),weight=self.weight2, bias=self.bias2, eps=1e-06)
        x83=x82.permute(0, 3, 1, 2)
        x84=self.conv2d8(x83)
        x86=self.conv2d9(x84)
        x87=torch.permute(x86, [0, 2, 3, 1])
        x88=self.layernorm6(x87)
        x89=self.linear12(x88)
        x90=self.gelu6(x89)
        x91=self.linear13(x90)
        x92=torch.permute(x91, [0, 3, 1, 2])
        x93=operator.mul(self.layer_scale6, x92)
        x94=stochastic_depth(x93, 0.03529411764705883, 'row', False)
        x95=operator.add(x94, x84)
        x97=self.conv2d10(x95)
        x98=torch.permute(x97, [0, 2, 3, 1])
        x99=self.layernorm7(x98)
        x100=self.linear14(x99)
        x101=self.gelu7(x100)
        x102=self.linear15(x101)
        x103=torch.permute(x102, [0, 3, 1, 2])
        x104=operator.mul(self.layer_scale7, x103)
        x105=stochastic_depth(x104, 0.0411764705882353, 'row', False)
        x106=operator.add(x105, x95)
        x108=self.conv2d11(x106)
        x109=torch.permute(x108, [0, 2, 3, 1])
        x110=self.layernorm8(x109)
        x111=self.linear16(x110)
        x112=self.gelu8(x111)
        x113=self.linear17(x112)
        x114=torch.permute(x113, [0, 3, 1, 2])
        x115=operator.mul(self.layer_scale8, x114)
        x116=stochastic_depth(x115, 0.047058823529411764, 'row', False)
        x117=operator.add(x116, x106)
        x119=self.conv2d12(x117)
        x120=torch.permute(x119, [0, 2, 3, 1])
        x121=self.layernorm9(x120)
        x122=self.linear18(x121)
        x123=self.gelu9(x122)
        x124=self.linear19(x123)
        x125=torch.permute(x124, [0, 3, 1, 2])
        x126=operator.mul(self.layer_scale9, x125)
        x127=stochastic_depth(x126, 0.052941176470588235, 'row', False)
        x128=operator.add(x127, x117)
        x130=self.conv2d13(x128)
        x131=torch.permute(x130, [0, 2, 3, 1])
        x132=self.layernorm10(x131)
        x133=self.linear20(x132)
        x134=self.gelu10(x133)
        x135=self.linear21(x134)
        x136=torch.permute(x135, [0, 3, 1, 2])
        x137=operator.mul(self.layer_scale10, x136)
        x138=stochastic_depth(x137, 0.058823529411764705, 'row', False)
        x139=operator.add(x138, x128)
        x141=self.conv2d14(x139)
        x142=torch.permute(x141, [0, 2, 3, 1])
        x143=self.layernorm11(x142)
        x144=self.linear22(x143)
        x145=self.gelu11(x144)
        x146=self.linear23(x145)
        x147=torch.permute(x146, [0, 3, 1, 2])
        x148=operator.mul(self.layer_scale11, x147)
        x149=stochastic_depth(x148, 0.06470588235294118, 'row', False)
        x150=operator.add(x149, x139)
        x152=self.conv2d15(x150)
        x153=torch.permute(x152, [0, 2, 3, 1])
        x154=self.layernorm12(x153)
        x155=self.linear24(x154)
        x156=self.gelu12(x155)
        x157=self.linear25(x156)
        x158=torch.permute(x157, [0, 3, 1, 2])
        x159=operator.mul(self.layer_scale12, x158)
        x160=stochastic_depth(x159, 0.07058823529411766, 'row', False)
        x161=operator.add(x160, x150)
        x163=self.conv2d16(x161)
        x164=torch.permute(x163, [0, 2, 3, 1])
        x165=self.layernorm13(x164)
        x166=self.linear26(x165)
        x167=self.gelu13(x166)
        x168=self.linear27(x167)
        x169=torch.permute(x168, [0, 3, 1, 2])
        x170=operator.mul(self.layer_scale13, x169)
        x171=stochastic_depth(x170, 0.07647058823529412, 'row', False)
        x172=operator.add(x171, x161)
        x174=self.conv2d17(x172)
        x175=torch.permute(x174, [0, 2, 3, 1])
        x176=self.layernorm14(x175)
        x177=self.linear28(x176)
        x178=self.gelu14(x177)
        x179=self.linear29(x178)
        x180=torch.permute(x179, [0, 3, 1, 2])
        x181=operator.mul(self.layer_scale14, x180)
        x182=stochastic_depth(x181, 0.0823529411764706, 'row', False)
        x183=operator.add(x182, x172)
        x184=x183.permute(0, 2, 3, 1)
        x187=torch.nn.functional.layer_norm(x184, (384,),weight=self.weight3, bias=self.bias3, eps=1e-06)
        x188=x187.permute(0, 3, 1, 2)
        x189=self.conv2d18(x188)
        x191=self.conv2d19(x189)
        x192=torch.permute(x191, [0, 2, 3, 1])
        x193=self.layernorm15(x192)
        x194=self.linear30(x193)
        x195=self.gelu15(x194)
        x196=self.linear31(x195)
        x197=torch.permute(x196, [0, 3, 1, 2])
        x198=operator.mul(self.layer_scale15, x197)
        x199=stochastic_depth(x198, 0.08823529411764706, 'row', False)
        x200=operator.add(x199, x189)
        x202=self.conv2d20(x200)
        x203=torch.permute(x202, [0, 2, 3, 1])
        x204=self.layernorm16(x203)
        x205=self.linear32(x204)
        x206=self.gelu16(x205)
        x207=self.linear33(x206)
        x208=torch.permute(x207, [0, 3, 1, 2])
        x209=operator.mul(self.layer_scale16, x208)
        x210=stochastic_depth(x209, 0.09411764705882353, 'row', False)
        x211=operator.add(x210, x200)
        x213=self.conv2d21(x211)
        x214=torch.permute(x213, [0, 2, 3, 1])
        x215=self.layernorm17(x214)
        x216=self.linear34(x215)
        x217=self.gelu17(x216)
        x218=self.linear35(x217)
        x219=torch.permute(x218, [0, 3, 1, 2])
        x220=operator.mul(self.layer_scale17, x219)
        x221=stochastic_depth(x220, 0.1, 'row', False)
        x222=operator.add(x221, x211)
        x223=self.adaptiveavgpool2d0(x222)
        x224=x223.permute(0, 2, 3, 1)
        x227=torch.nn.functional.layer_norm(x224, (768,),weight=self.weight4, bias=self.bias4, eps=1e-06)
        x228=x227.permute(0, 3, 1, 2)
        x229=self.flatten0(x228)
        x230=self.linear36(x229)

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.randn(batch_size, 3, 224, 224)
start_time=time.time()
for i in range(10):
    output = m(x)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
