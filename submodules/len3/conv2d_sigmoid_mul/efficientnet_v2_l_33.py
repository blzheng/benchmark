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
        self.conv2d201 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid33 = Sigmoid()

    def forward(self, x648, x645):
        x649=self.conv2d201(x648)
        x650=self.sigmoid33(x649)
        x651=operator.mul(x650, x645)
        return x651

m = M().eval()
x648 = torch.randn(torch.Size([1, 96, 1, 1]))
x645 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x648, x645)
end = time.time()
print(end-start)
