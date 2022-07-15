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
        self.conv2d6 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x16):
        x17=self.conv2d6(x16)
        return x17

m = M().eval()
x16 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)
