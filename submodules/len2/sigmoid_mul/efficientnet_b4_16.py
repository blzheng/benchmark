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
        self.sigmoid16 = Sigmoid()

    def forward(self, x255, x251):
        x256=self.sigmoid16(x255)
        x257=operator.mul(x256, x251)
        return x257

m = M().eval()
x255 = torch.randn(torch.Size([1, 672, 1, 1]))
x251 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x255, x251)
end = time.time()
print(end-start)
