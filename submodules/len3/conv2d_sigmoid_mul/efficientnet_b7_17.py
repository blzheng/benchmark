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
        self.conv2d85 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()

    def forward(self, x266, x263):
        x267=self.conv2d85(x266)
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        return x269

m = M().eval()
x266 = torch.randn(torch.Size([1, 20, 1, 1]))
x263 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x266, x263)
end = time.time()
print(end-start)
