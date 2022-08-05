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
        self.conv2d10 = Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)

    def forward(self, x29):
        x30=self.conv2d10(x29)
        return x30

m = M().eval()
x29 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
