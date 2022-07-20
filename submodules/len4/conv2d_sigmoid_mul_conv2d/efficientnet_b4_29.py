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
        self.conv2d147 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()
        self.conv2d148 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x458, x455):
        x459=self.conv2d147(x458)
        x460=self.sigmoid29(x459)
        x461=operator.mul(x460, x455)
        x462=self.conv2d148(x461)
        return x462

m = M().eval()
x458 = torch.randn(torch.Size([1, 68, 1, 1]))
x455 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x458, x455)
end = time.time()
print(end-start)
