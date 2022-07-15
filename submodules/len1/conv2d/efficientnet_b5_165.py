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
        self.conv2d165 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x517):
        x518=self.conv2d165(x517)
        return x518

m = M().eval()
x517 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x517)
end = time.time()
print(end-start)
