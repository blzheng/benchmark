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
        self.conv2d14 = Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        self.relu14 = ReLU(inplace=True)

    def forward(self, x34):
        x35=self.conv2d14(x34)
        x36=self.relu14(x35)
        return x36

m = M().eval()
x34 = torch.randn(torch.Size([1, 48, 27, 27]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)