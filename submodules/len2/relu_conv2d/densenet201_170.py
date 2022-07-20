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
        self.relu171 = ReLU(inplace=True)
        self.conv2d171 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x605):
        x606=self.relu171(x605)
        x607=self.conv2d171(x606)
        return x607

m = M().eval()
x605 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x605)
end = time.time()
print(end-start)
