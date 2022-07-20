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
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x600, x595):
        x601=operator.mul(x600, x595)
        x602=self.conv2d191(x601)
        x603=self.batchnorm2d113(x602)
        return x603

m = M().eval()
x600 = torch.randn(torch.Size([1, 1344, 1, 1]))
x595 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x600, x595)
end = time.time()
print(end-start)
