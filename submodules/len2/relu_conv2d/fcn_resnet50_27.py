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
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x104):
        x105=self.relu28(x104)
        x106=self.conv2d32(x105)
        return x106

m = M().eval()
x104 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x104)
end = time.time()
print(end-start)
