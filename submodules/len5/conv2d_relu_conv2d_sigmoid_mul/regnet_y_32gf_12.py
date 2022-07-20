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
        self.conv2d66 = Conv2d(1392, 348, kernel_size=(1, 1), stride=(1, 1))
        self.relu51 = ReLU()
        self.conv2d67 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x208, x207):
        x209=self.conv2d66(x208)
        x210=self.relu51(x209)
        x211=self.conv2d67(x210)
        x212=self.sigmoid12(x211)
        x213=operator.mul(x212, x207)
        return x213

m = M().eval()
x208 = torch.randn(torch.Size([1, 1392, 1, 1]))
x207 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x208, x207)
end = time.time()
print(end-start)
