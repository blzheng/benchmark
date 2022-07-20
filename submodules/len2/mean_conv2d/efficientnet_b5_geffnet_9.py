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

    def forward(self, x135):
        x136=x135.mean((2, 3),keepdim=True)
        x137=self.conv2d45(x136)
        return x137

m = M().eval()
x135 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)
