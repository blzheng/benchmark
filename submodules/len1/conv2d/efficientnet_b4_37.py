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
        self.conv2d37 = Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x112):
        x113=self.conv2d37(x112)
        return x113

m = M().eval()
x112 = torch.randn(torch.Size([1, 14, 1, 1]))
start = time.time()
output = m(x112)
end = time.time()
print(end-start)
