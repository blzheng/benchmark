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
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x286, x281):
        x287=operator.mul(x286, x281)
        x288=self.conv2d93(x287)
        return x288

m = M().eval()
x286 = torch.randn(torch.Size([1, 816, 1, 1]))
x281 = torch.randn(torch.Size([1, 816, 7, 7]))
start = time.time()
output = m(x286, x281)
end = time.time()
print(end-start)
