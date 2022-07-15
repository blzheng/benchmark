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
        self.conv2d72 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)

    def forward(self, x234):
        x235=self.conv2d72(x234)
        return x235

m = M().eval()
x234 = torch.randn(torch.Size([1, 1920, 14, 14]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
