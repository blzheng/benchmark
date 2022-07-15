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
        self.conv2d215 = Conv2d(1824, 1824, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1824, bias=False)

    def forward(self, x689):
        x690=self.conv2d215(x689)
        return x690

m = M().eval()
x689 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x689)
end = time.time()
print(end-start)
