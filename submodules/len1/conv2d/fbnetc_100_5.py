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
        self.conv2d5 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)

    def forward(self, x16):
        x17=self.conv2d5(x16)
        return x17

m = M().eval()
x16 = torch.randn(torch.Size([1, 96, 112, 112]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)
