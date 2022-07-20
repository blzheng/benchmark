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
        self.conv2d23 = Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu8 = ReLU()
        self.conv2d24 = Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x67):
        x68=self.conv2d23(x67)
        x69=self.relu8(x68)
        x70=self.conv2d24(x69)
        return x70

m = M().eval()
x67 = torch.randn(torch.Size([1, 240, 1, 1]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
