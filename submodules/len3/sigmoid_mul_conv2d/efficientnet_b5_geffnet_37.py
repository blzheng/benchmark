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
        self.conv2d187 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x555, x551):
        x556=x555.sigmoid()
        x557=operator.mul(x551, x556)
        x558=self.conv2d187(x557)
        return x558

m = M().eval()
x555 = torch.randn(torch.Size([1, 3072, 1, 1]))
x551 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x555, x551)
end = time.time()
print(end-start)
