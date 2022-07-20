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
        self.sigmoid27 = Sigmoid()
        self.conv2d158 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x501, x497):
        x502=self.sigmoid27(x501)
        x503=operator.mul(x502, x497)
        x504=self.conv2d158(x503)
        return x504

m = M().eval()
x501 = torch.randn(torch.Size([1, 1536, 1, 1]))
x497 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x501, x497)
end = time.time()
print(end-start)
