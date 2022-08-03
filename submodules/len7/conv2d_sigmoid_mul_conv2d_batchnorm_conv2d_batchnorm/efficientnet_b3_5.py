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
        self.conv2d92 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x284, x281):
        x285=self.conv2d92(x284)
        x286=self.sigmoid18(x285)
        x287=operator.mul(x286, x281)
        x288=self.conv2d93(x287)
        x289=self.batchnorm2d55(x288)
        x290=self.conv2d94(x289)
        x291=self.batchnorm2d56(x290)
        return x291

m = M().eval()
x284 = torch.randn(torch.Size([1, 34, 1, 1]))
x281 = torch.randn(torch.Size([1, 816, 7, 7]))
start = time.time()
output = m(x284, x281)
end = time.time()
print(end-start)
