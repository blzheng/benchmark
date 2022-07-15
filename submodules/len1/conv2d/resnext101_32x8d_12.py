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
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)

    def forward(self, x39):
        x40=self.conv2d12(x39)
        return x40

m = M().eval()
x39 = torch.randn(torch.Size([1, 512, 56, 56]))
start = time.time()
output = m(x39)
end = time.time()
print(end-start)
