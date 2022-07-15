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
        self.conv2d157 = Conv2d(112, 2688, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x488):
        x489=self.conv2d157(x488)
        return x489

m = M().eval()
x488 = torch.randn(torch.Size([1, 112, 1, 1]))
start = time.time()
output = m(x488)
end = time.time()
print(end-start)
