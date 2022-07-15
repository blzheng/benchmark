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
        self.conv2d5 = Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x23):
        x24=self.conv2d5(x23)
        return x24

m = M().eval()
x23 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
