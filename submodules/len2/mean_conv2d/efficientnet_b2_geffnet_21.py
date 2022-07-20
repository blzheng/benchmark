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
        self.conv2d106 = Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x314):
        x315=x314.mean((2, 3),keepdim=True)
        x316=self.conv2d106(x315)
        return x316

m = M().eval()
x314 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x314)
end = time.time()
print(end-start)
