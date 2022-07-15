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
        self.conv2d60 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x196):
        x197=self.conv2d60(x196)
        return x197

m = M().eval()
x196 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
