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
        self.conv2d95 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)

    def forward(self, x292):
        x293=self.conv2d95(x292)
        return x293

m = M().eval()
x292 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x292)
end = time.time()
print(end-start)
