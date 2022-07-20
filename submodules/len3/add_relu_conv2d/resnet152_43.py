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
        self.relu130 = ReLU(inplace=True)
        self.conv2d136 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x448, x440):
        x449=operator.add(x448, x440)
        x450=self.relu130(x449)
        x451=self.conv2d136(x450)
        return x451

m = M().eval()
x448 = torch.randn(torch.Size([1, 1024, 14, 14]))
x440 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x448, x440)
end = time.time()
print(end-start)
