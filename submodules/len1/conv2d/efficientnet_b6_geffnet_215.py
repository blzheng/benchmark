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
        self.conv2d215 = Conv2d(3456, 144, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x642):
        x643=self.conv2d215(x642)
        return x643

m = M().eval()
x642 = torch.randn(torch.Size([1, 3456, 1, 1]))
start = time.time()
output = m(x642)
end = time.time()
print(end-start)
