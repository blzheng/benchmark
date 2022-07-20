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
        self.conv2d181 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()

    def forward(self, x586):
        x587=self.conv2d181(x586)
        x588=self.sigmoid29(x587)
        return x588

m = M().eval()
x586 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x586)
end = time.time()
print(end-start)
