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
        self.conv2d44 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x145):
        x146=self.conv2d44(x145)
        return x146

m = M().eval()
x145 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
