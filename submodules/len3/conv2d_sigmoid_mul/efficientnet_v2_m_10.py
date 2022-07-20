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
        self.conv2d77 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x249, x246):
        x250=self.conv2d77(x249)
        x251=self.sigmoid10(x250)
        x252=operator.mul(x251, x246)
        return x252

m = M().eval()
x249 = torch.randn(torch.Size([1, 44, 1, 1]))
x246 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x249, x246)
end = time.time()
print(end-start)
