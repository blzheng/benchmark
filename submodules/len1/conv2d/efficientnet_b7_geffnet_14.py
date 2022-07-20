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
        self.conv2d14 = Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x45):
        x46=self.conv2d14(x45)
        return x46

m = M().eval()
x45 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)