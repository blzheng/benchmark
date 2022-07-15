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
        self.conv2d86 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x283):
        x284=self.conv2d86(x283)
        return x284

m = M().eval()
x283 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x283)
end = time.time()
print(end-start)
