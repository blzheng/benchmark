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
        self.conv2d95 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()

    def forward(self, x296, x293):
        x297=self.conv2d95(x296)
        x298=self.sigmoid19(x297)
        x299=operator.mul(x298, x293)
        return x299

m = M().eval()
x296 = torch.randn(torch.Size([1, 40, 1, 1]))
x293 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x296, x293)
end = time.time()
print(end-start)
