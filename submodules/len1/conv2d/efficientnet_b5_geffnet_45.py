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
        self.conv2d45 = Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x136):
        x137=self.conv2d45(x136)
        return x137

m = M().eval()
x136 = torch.randn(torch.Size([1, 384, 1, 1]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
