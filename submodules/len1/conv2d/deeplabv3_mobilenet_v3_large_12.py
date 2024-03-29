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
        self.conv2d12 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x37):
        x38=self.conv2d12(x37)
        return x38

m = M().eval()
x37 = torch.randn(torch.Size([1, 24, 1, 1]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
