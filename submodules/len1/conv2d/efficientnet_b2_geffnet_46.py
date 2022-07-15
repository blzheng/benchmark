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
        self.conv2d46 = Conv2d(528, 22, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x137):
        x138=self.conv2d46(x137)
        return x138

m = M().eval()
x137 = torch.randn(torch.Size([1, 528, 1, 1]))
start = time.time()
output = m(x137)
end = time.time()
print(end-start)
