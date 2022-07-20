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
        self.relu33 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(864, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x143):
        x144=self.relu33(x143)
        x145=self.conv2d50(x144)
        return x145

m = M().eval()
x143 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
