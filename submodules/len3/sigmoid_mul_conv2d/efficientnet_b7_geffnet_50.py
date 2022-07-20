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
        self.conv2d251 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x749, x745):
        x750=x749.sigmoid()
        x751=operator.mul(x745, x750)
        x752=self.conv2d251(x751)
        return x752

m = M().eval()
x749 = torch.randn(torch.Size([1, 2304, 1, 1]))
x745 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x749, x745)
end = time.time()
print(end-start)
