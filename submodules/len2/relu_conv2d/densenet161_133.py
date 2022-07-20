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
        self.relu134 = ReLU(inplace=True)
        self.conv2d134 = Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x476):
        x477=self.relu134(x476)
        x478=self.conv2d134(x477)
        return x478

m = M().eval()
x476 = torch.randn(torch.Size([1, 1584, 7, 7]))
start = time.time()
output = m(x476)
end = time.time()
print(end-start)
