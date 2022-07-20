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
        self.batchnorm2d66 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x217, x210):
        x218=self.batchnorm2d66(x217)
        x219=operator.add(x218, x210)
        x220=self.relu61(x219)
        x221=self.conv2d67(x220)
        x222=self.batchnorm2d67(x221)
        return x222

m = M().eval()
x217 = torch.randn(torch.Size([1, 1024, 14, 14]))
x210 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x217, x210)
end = time.time()
print(end-start)
