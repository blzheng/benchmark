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
        self.relu198 = ReLU(inplace=True)
        self.conv2d198 = Conv2d(1888, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x700):
        x701=self.relu198(x700)
        x702=self.conv2d198(x701)
        return x702

m = M().eval()
x700 = torch.randn(torch.Size([1, 1888, 7, 7]))
start = time.time()
output = m(x700)
end = time.time()
print(end-start)
