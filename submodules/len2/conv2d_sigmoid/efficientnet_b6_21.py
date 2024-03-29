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
        self.conv2d106 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()

    def forward(self, x331):
        x332=self.conv2d106(x331)
        x333=self.sigmoid21(x332)
        return x333

m = M().eval()
x331 = torch.randn(torch.Size([1, 36, 1, 1]))
start = time.time()
output = m(x331)
end = time.time()
print(end-start)
