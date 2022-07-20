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
        self.conv2d25 = Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x88, x82):
        x89=operator.add(x88, x82)
        x90=self.conv2d25(x89)
        return x90

m = M().eval()
x88 = torch.randn(torch.Size([1, 96, 28, 28]))
x82 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x88, x82)
end = time.time()
print(end-start)
