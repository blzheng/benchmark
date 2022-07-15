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
        self.conv2d80 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x263):
        x264=self.conv2d80(x263)
        return x264

m = M().eval()
x263 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
