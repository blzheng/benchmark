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
        self.conv2d61 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x201):
        x202=self.conv2d61(x201)
        return x202

m = M().eval()
x201 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x201)
end = time.time()
print(end-start)
