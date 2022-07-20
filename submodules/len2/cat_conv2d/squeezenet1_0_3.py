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
        self.conv2d16 = Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x36, x38):
        x39=torch.cat([x36, x38], 1)
        x40=self.conv2d16(x39)
        return x40

m = M().eval()
x36 = torch.randn(torch.Size([1, 192, 27, 27]))
x38 = torch.randn(torch.Size([1, 192, 27, 27]))
start = time.time()
output = m(x36, x38)
end = time.time()
print(end-start)
