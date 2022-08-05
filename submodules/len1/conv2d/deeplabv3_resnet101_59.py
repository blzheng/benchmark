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
        self.conv2d59 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x195):
        x196=self.conv2d59(x195)
        return x196

m = M().eval()
x195 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)
