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
        self.sigmoid54 = Sigmoid()
        self.conv2d307 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x985, x981):
        x986=self.sigmoid54(x985)
        x987=operator.mul(x986, x981)
        x988=self.conv2d307(x987)
        return x988

m = M().eval()
x985 = torch.randn(torch.Size([1, 2304, 1, 1]))
x981 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x985, x981)
end = time.time()
print(end-start)
