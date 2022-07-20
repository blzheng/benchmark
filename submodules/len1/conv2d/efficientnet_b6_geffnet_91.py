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
        self.conv2d91 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x272):
        x273=self.conv2d91(x272)
        return x273

m = M().eval()
x272 = torch.randn(torch.Size([1, 36, 1, 1]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)