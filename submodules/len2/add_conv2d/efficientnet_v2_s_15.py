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
        self.conv2d69 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x220, x205):
        x221=operator.add(x220, x205)
        x222=self.conv2d69(x221)
        return x222

m = M().eval()
x220 = torch.randn(torch.Size([1, 160, 14, 14]))
x205 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x220, x205)
end = time.time()
print(end-start)
