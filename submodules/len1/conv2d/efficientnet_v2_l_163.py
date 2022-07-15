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
        self.conv2d163 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x529):
        x530=self.conv2d163(x529)
        return x530

m = M().eval()
x529 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x529)
end = time.time()
print(end-start)
