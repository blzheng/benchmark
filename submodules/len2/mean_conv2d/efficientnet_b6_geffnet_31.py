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
        self.conv2d155 = Conv2d(1200, 50, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x463):
        x464=x463.mean((2, 3),keepdim=True)
        x465=self.conv2d155(x464)
        return x465

m = M().eval()
x463 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x463)
end = time.time()
print(end-start)
