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
        self.conv2d107 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x332):
        x333=self.conv2d107(x332)
        return x333

m = M().eval()
x332 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x332)
end = time.time()
print(end-start)
