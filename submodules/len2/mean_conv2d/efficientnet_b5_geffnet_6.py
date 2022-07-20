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
        self.conv2d30 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x91):
        x92=x91.mean((2, 3),keepdim=True)
        x93=self.conv2d30(x92)
        return x93

m = M().eval()
x91 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
