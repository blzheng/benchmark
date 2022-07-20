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
        self.conv2d231 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid39 = Sigmoid()
        self.conv2d232 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x744, x741):
        x745=self.conv2d231(x744)
        x746=self.sigmoid39(x745)
        x747=operator.mul(x746, x741)
        x748=self.conv2d232(x747)
        return x748

m = M().eval()
x744 = torch.randn(torch.Size([1, 96, 1, 1]))
x741 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x744, x741)
end = time.time()
print(end-start)
