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
        self.conv2d36 = Conv2d(1232, 112, kernel_size=(1, 1), stride=(1, 1))
        self.relu27 = ReLU()
        self.conv2d37 = Conv2d(112, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d38 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x111):
        x113=self.conv2d36(x112)
        x114=self.relu27(x113)
        x115=self.conv2d37(x114)
        x116=self.sigmoid6(x115)
        x117=operator.mul(x116, x111)
        x118=self.conv2d38(x117)
        return x118

m = M().eval()
x112 = torch.randn(torch.Size([1, 1232, 1, 1]))
x111 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x112, x111)
end = time.time()
print(end-start)
