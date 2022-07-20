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
        self.conv2d143 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x464, x449):
        x465=operator.add(x464, x449)
        x466=self.conv2d143(x465)
        return x466

m = M().eval()
x464 = torch.randn(torch.Size([1, 224, 14, 14]))
x449 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x464, x449)
end = time.time()
print(end-start)
