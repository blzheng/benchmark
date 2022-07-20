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
        self.relu68 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x280):
        x281=self.relu68(x280)
        x282=self.conv2d89(x281)
        return x282

m = M().eval()
x280 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
