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
        self.conv2d225 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x726):
        x727=self.conv2d225(x726)
        return x727

m = M().eval()
x726 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x726)
end = time.time()
print(end-start)
