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
        self.conv2d97 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x313, x310):
        x314=self.conv2d97(x313)
        x315=self.sigmoid14(x314)
        x316=operator.mul(x315, x310)
        return x316

m = M().eval()
x313 = torch.randn(torch.Size([1, 44, 1, 1]))
x310 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x313, x310)
end = time.time()
print(end-start)
