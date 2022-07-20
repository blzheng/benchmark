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
        self.conv2d142 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()
        self.conv2d143 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x442, x439):
        x443=self.conv2d142(x442)
        x444=self.sigmoid28(x443)
        x445=operator.mul(x444, x439)
        x446=self.conv2d143(x445)
        x447=self.batchnorm2d85(x446)
        return x447

m = M().eval()
x442 = torch.randn(torch.Size([1, 68, 1, 1]))
x439 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x442, x439)
end = time.time()
print(end-start)
