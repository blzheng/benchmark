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
        self.conv2d13 = Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x29, x31):
        x32=torch.cat([x29, x31], 1)
        x33=self.conv2d13(x32)
        return x33

m = M().eval()
x29 = torch.randn(torch.Size([1, 128, 27, 27]))
x31 = torch.randn(torch.Size([1, 128, 27, 27]))
start = time.time()
output = m(x29, x31)
end = time.time()
print(end-start)
