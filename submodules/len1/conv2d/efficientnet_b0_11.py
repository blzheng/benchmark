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
        self.conv2d11 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)

    def forward(self, x31):
        x32=self.conv2d11(x31)
        return x32

m = M().eval()
x31 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x31)
end = time.time()
print(end-start)
