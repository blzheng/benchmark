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
        self.conv2d161 = Conv2d(86, 2064, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x480):
        x481=self.conv2d161(x480)
        return x481

m = M().eval()
x480 = torch.randn(torch.Size([1, 86, 1, 1]))
start = time.time()
output = m(x480)
end = time.time()
print(end-start)
