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
        self.sigmoid20 = Sigmoid()
        self.conv2d101 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x313, x309):
        x314=self.sigmoid20(x313)
        x315=operator.mul(x314, x309)
        x316=self.conv2d101(x315)
        return x316

m = M().eval()
x313 = torch.randn(torch.Size([1, 960, 1, 1]))
x309 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x313, x309)
end = time.time()
print(end-start)
