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
        self.conv2d74 = Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x253):
        x254=self.conv2d74(x253)
        return x254

m = M().eval()
x253 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
