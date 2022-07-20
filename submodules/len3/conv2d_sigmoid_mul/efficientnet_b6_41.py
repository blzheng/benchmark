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
        self.conv2d206 = Conv2d(86, 2064, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid41 = Sigmoid()

    def forward(self, x647, x644):
        x648=self.conv2d206(x647)
        x649=self.sigmoid41(x648)
        x650=operator.mul(x649, x644)
        return x650

m = M().eval()
x647 = torch.randn(torch.Size([1, 86, 1, 1]))
x644 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x647, x644)
end = time.time()
print(end-start)
