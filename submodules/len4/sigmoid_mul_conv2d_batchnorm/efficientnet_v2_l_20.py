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
        self.sigmoid20 = Sigmoid()
        self.conv2d137 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x443, x439):
        x444=self.sigmoid20(x443)
        x445=operator.mul(x444, x439)
        x446=self.conv2d137(x445)
        x447=self.batchnorm2d95(x446)
        return x447

m = M().eval()
x443 = torch.randn(torch.Size([1, 1344, 1, 1]))
x439 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x443, x439)
end = time.time()
print(end-start)
