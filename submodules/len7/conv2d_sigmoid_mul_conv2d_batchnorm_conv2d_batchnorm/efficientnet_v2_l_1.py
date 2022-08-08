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
        self.conv2d86 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d87 = Conv2d(1152, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d88 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x284, x281):
        x285=self.conv2d86(x284)
        x286=self.sigmoid10(x285)
        x287=operator.mul(x286, x281)
        x288=self.conv2d87(x287)
        x289=self.batchnorm2d65(x288)
        x290=self.conv2d88(x289)
        x291=self.batchnorm2d66(x290)
        return x291

m = M().eval()
x284 = torch.randn(torch.Size([1, 48, 1, 1]))
x281 = torch.randn(torch.Size([1, 1152, 14, 14]))
start = time.time()
output = m(x284, x281)
end = time.time()
print(end-start)
