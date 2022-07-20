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
        self.conv2d68 = Conv2d(576, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d69 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x196, x201):
        x202=operator.mul(x196, x201)
        x203=self.conv2d68(x202)
        x204=self.batchnorm2d40(x203)
        x205=self.conv2d69(x204)
        return x205

m = M().eval()
x196 = torch.randn(torch.Size([1, 576, 14, 14]))
x201 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x196, x201)
end = time.time()
print(end-start)
