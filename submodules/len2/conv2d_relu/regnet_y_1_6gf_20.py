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
        self.conv2d106 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu83 = ReLU()

    def forward(self, x336):
        x337=self.conv2d106(x336)
        x338=self.relu83(x337)
        return x338

m = M().eval()
x336 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
