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
        self.sigmoid18 = Sigmoid()
        self.conv2d98 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x307, x303):
        x308=self.sigmoid18(x307)
        x309=operator.mul(x308, x303)
        x310=self.conv2d98(x309)
        return x310

m = M().eval()
x307 = torch.randn(torch.Size([1, 2904, 1, 1]))
x303 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x307, x303)
end = time.time()
print(end-start)
