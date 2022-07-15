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
        self.conv2d104 = Conv2d(3712, 3712, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x327):
        x328=self.conv2d104(x327)
        return x328

m = M().eval()
x327 = torch.randn(torch.Size([1, 3712, 7, 7]))
start = time.time()
output = m(x327)
end = time.time()
print(end-start)
