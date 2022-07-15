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
        self.conv2d36 = Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x108):
        x109=self.conv2d36(x108)
        return x109

m = M().eval()
x108 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
