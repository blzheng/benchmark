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
        self.conv2d126 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x410):
        x411=self.conv2d126(x410)
        return x411

m = M().eval()
x410 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x410)
end = time.time()
print(end-start)
