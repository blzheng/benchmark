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
        self.relu7 = ReLU()
        self.conv2d19 = Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid2 = Hardsigmoid()

    def forward(self, x53):
        x54=self.relu7(x53)
        x55=self.conv2d19(x54)
        x56=self.hardsigmoid2(x55)
        return x56

m = M().eval()
x53 = torch.randn(torch.Size([1, 64, 1, 1]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
