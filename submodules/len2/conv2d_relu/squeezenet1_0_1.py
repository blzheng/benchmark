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
        self.conv2d1 = Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = ReLU(inplace=True)

    def forward(self, x3):
        x4=self.conv2d1(x3)
        x5=self.relu1(x4)
        return x5

m = M().eval()
x3 = torch.randn(torch.Size([1, 96, 54, 54]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
