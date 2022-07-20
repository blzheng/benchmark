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
        self.sigmoid38 = Sigmoid()
        self.conv2d227 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x729, x725):
        x730=self.sigmoid38(x729)
        x731=operator.mul(x730, x725)
        x732=self.conv2d227(x731)
        return x732

m = M().eval()
x729 = torch.randn(torch.Size([1, 2304, 1, 1]))
x725 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x729, x725)
end = time.time()
print(end-start)
