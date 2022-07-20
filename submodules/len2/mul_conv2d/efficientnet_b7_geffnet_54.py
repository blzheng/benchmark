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
        self.conv2d271 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x804, x809):
        x810=operator.mul(x804, x809)
        x811=self.conv2d271(x810)
        return x811

m = M().eval()
x804 = torch.randn(torch.Size([1, 3840, 7, 7]))
x809 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x804, x809)
end = time.time()
print(end-start)
