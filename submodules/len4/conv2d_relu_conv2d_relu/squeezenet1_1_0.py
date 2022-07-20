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
        self.conv2d1 = Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu2 = ReLU(inplace=True)

    def forward(self, x3):
        x4=self.conv2d1(x3)
        x5=self.relu1(x4)
        x6=self.conv2d2(x5)
        x7=self.relu2(x6)
        return x7

m = M().eval()
x3 = torch.randn(torch.Size([1, 64, 55, 55]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
