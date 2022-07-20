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
        self.conv2d96 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x285):
        x286=x285.mean((2, 3),keepdim=True)
        x287=self.conv2d96(x286)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
