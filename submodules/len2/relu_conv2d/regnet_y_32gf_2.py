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
        self.relu3 = ReLU()
        self.conv2d5 = Conv2d(8, 232, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x13):
        x14=self.relu3(x13)
        x15=self.conv2d5(x14)
        return x15

m = M().eval()
x13 = torch.randn(torch.Size([1, 8, 1, 1]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
