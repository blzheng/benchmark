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
        self.conv2d53 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x174):
        x175=self.conv2d53(x174)
        return x175

m = M().eval()
x174 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
