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
        self.conv2d194 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x610):
        x611=self.conv2d194(x610)
        return x611

m = M().eval()
x610 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x610)
end = time.time()
print(end-start)
