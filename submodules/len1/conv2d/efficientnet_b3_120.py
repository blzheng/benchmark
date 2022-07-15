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
        self.conv2d120 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1392, bias=False)

    def forward(self, x372):
        x373=self.conv2d120(x372)
        return x373

m = M().eval()
x372 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x372)
end = time.time()
print(end-start)
