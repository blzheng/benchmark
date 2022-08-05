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
        self.conv2d11 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
        self.relu7 = ReLU()
        self.conv2d12 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x33):
        x34=self.conv2d11(x33)
        x35=self.relu7(x34)
        x36=self.conv2d12(x35)
        return x36

m = M().eval()
x33 = torch.randn(torch.Size([1, 72, 1, 1]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
