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
        self.conv2d83 = Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)

    def forward(self, x280):
        x284=self.conv2d83(x280)
        return x284

m = M().eval()
x280 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
