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
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x227, x220):
        x228=self.batchnorm2d69(x227)
        x229=operator.add(x228, x220)
        x230=self.relu64(x229)
        x231=self.conv2d70(x230)
        return x231

m = M().eval()
x227 = torch.randn(torch.Size([1, 1024, 14, 14]))
x220 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x227, x220)
end = time.time()
print(end-start)
