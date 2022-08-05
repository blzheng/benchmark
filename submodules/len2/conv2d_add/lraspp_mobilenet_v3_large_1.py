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
        self.conv2d65 = Conv2d(128, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x191, x192):
        x193=self.conv2d65(x191)
        x194=operator.add(x192, x193)
        return x194

m = M().eval()
x191 = torch.randn(torch.Size([1, 128, 28, 28]))
x192 = torch.randn(torch.Size([1, 21, 28, 28]))
start = time.time()
output = m(x191, x192)
end = time.time()
print(end-start)
