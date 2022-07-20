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
        self.conv2d98 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x284, x289):
        x290=operator.mul(x284, x289)
        x291=self.conv2d98(x290)
        x292=self.batchnorm2d58(x291)
        return x292

m = M().eval()
x284 = torch.randn(torch.Size([1, 1248, 7, 7]))
x289 = torch.randn(torch.Size([1, 1248, 1, 1]))
start = time.time()
output = m(x284, x289)
end = time.time()
print(end-start)
