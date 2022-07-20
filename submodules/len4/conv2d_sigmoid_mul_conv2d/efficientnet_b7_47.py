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
        self.conv2d235 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid47 = Sigmoid()
        self.conv2d236 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x740, x737):
        x741=self.conv2d235(x740)
        x742=self.sigmoid47(x741)
        x743=operator.mul(x742, x737)
        x744=self.conv2d236(x743)
        return x744

m = M().eval()
x740 = torch.randn(torch.Size([1, 96, 1, 1]))
x737 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x740, x737)
end = time.time()
print(end-start)
