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
        self.relu9 = ReLU()
        self.conv2d29 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x83):
        x84=self.relu9(x83)
        x85=self.conv2d29(x84)
        return x85

m = M().eval()
x83 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x83)
end = time.time()
print(end-start)
