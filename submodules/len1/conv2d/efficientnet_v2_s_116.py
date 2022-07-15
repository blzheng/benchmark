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
        self.conv2d116 = Conv2d(1536, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x370):
        x371=self.conv2d116(x370)
        return x371

m = M().eval()
x370 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x370)
end = time.time()
print(end-start)
