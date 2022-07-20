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
        self.sigmoid26 = Sigmoid()
        self.conv2d153 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x485, x481):
        x486=self.sigmoid26(x485)
        x487=operator.mul(x486, x481)
        x488=self.conv2d153(x487)
        return x488

m = M().eval()
x485 = torch.randn(torch.Size([1, 1536, 1, 1]))
x481 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x485, x481)
end = time.time()
print(end-start)
