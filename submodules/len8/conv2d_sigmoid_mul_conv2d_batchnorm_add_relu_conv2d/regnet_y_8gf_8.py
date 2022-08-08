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

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d47 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d48 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x146, x143, x137):
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        x150=self.conv2d48(x149)
        x151=self.batchnorm2d30(x150)
        x152=operator.add(x137, x151)
        x153=self.relu36(x152)
        x154=self.conv2d49(x153)
        return x154

m = M().eval()
x146 = torch.randn(torch.Size([1, 224, 1, 1]))
x143 = torch.randn(torch.Size([1, 896, 14, 14]))
x137 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x146, x143, x137)
end = time.time()
print(end-start)
