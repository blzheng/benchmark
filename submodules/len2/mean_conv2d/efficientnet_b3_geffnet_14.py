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
        self.conv2d71 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x210):
        x211=x210.mean((2, 3),keepdim=True)
        x212=self.conv2d71(x211)
        return x212

m = M().eval()
x210 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
