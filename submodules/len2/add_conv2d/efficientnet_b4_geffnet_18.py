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
        self.conv2d124 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x367, x353):
        x368=operator.add(x367, x353)
        x369=self.conv2d124(x368)
        return x369

m = M().eval()
x367 = torch.randn(torch.Size([1, 272, 7, 7]))
x353 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x367, x353)
end = time.time()
print(end-start)
