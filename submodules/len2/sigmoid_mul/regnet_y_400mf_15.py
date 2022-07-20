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
        self.sigmoid15 = Sigmoid()

    def forward(self, x261, x257):
        x262=self.sigmoid15(x261)
        x263=operator.mul(x262, x257)
        return x263

m = M().eval()
x261 = torch.randn(torch.Size([1, 440, 1, 1]))
x257 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x261, x257)
end = time.time()
print(end-start)
