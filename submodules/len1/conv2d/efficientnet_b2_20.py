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
        self.conv2d20 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)

    def forward(self, x60):
        x61=self.conv2d20(x60)
        return x61

m = M().eval()
x60 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
