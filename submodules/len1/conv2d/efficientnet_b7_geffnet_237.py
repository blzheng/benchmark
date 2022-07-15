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
        self.conv2d237 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x709):
        x710=self.conv2d237(x709)
        return x710

m = M().eval()
x709 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x709)
end = time.time()
print(end-start)
