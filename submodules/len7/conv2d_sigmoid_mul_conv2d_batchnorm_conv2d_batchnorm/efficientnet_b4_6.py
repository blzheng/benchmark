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
        self.conv2d152 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()
        self.conv2d153 = Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d154 = Conv2d(448, 2688, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d92 = BatchNorm2d(2688, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x474, x471):
        x475=self.conv2d152(x474)
        x476=self.sigmoid30(x475)
        x477=operator.mul(x476, x471)
        x478=self.conv2d153(x477)
        x479=self.batchnorm2d91(x478)
        x480=self.conv2d154(x479)
        x481=self.batchnorm2d92(x480)
        return x481

m = M().eval()
x474 = torch.randn(torch.Size([1, 68, 1, 1]))
x471 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x474, x471)
end = time.time()
print(end-start)
