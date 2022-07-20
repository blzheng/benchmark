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
        self.conv2d36 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x107):
        x108=x107.mean((2, 3),keepdim=True)
        x109=self.conv2d36(x108)
        return x109

m = M().eval()
x107 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)
