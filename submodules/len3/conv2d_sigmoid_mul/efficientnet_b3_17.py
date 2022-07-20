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
        self.conv2d87 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()

    def forward(self, x268, x265):
        x269=self.conv2d87(x268)
        x270=self.sigmoid17(x269)
        x271=operator.mul(x270, x265)
        return x271

m = M().eval()
x268 = torch.randn(torch.Size([1, 34, 1, 1]))
x265 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x268, x265)
end = time.time()
print(end-start)
