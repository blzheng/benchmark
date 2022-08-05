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
        self.conv2d48 = Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
        self.relu16 = ReLU()
        self.conv2d49 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x143):
        x144=self.conv2d48(x143)
        x145=self.relu16(x144)
        x146=self.conv2d49(x145)
        return x146

m = M().eval()
x143 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
