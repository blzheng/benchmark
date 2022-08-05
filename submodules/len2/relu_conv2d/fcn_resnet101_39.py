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
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x144):
        x145=self.relu40(x144)
        x146=self.conv2d44(x145)
        return x146

m = M().eval()
x144 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
