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
        self.conv2d146 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x431, x436):
        x437=operator.mul(x431, x436)
        x438=self.conv2d146(x437)
        return x438

m = M().eval()
x431 = torch.randn(torch.Size([1, 1344, 14, 14]))
x436 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x431, x436)
end = time.time()
print(end-start)
