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
        self.relu36 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x121, x129):
        x130=operator.add(x121, x129)
        x131=self.relu36(x130)
        x132=self.conv2d41(x131)
        return x132

m = M().eval()
x121 = torch.randn(torch.Size([1, 400, 7, 7]))
x129 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x121, x129)
end = time.time()
print(end-start)
