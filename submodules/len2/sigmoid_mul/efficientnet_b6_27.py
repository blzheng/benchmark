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
        self.sigmoid27 = Sigmoid()

    def forward(self, x426, x422):
        x427=self.sigmoid27(x426)
        x428=operator.mul(x427, x422)
        return x428

m = M().eval()
x426 = torch.randn(torch.Size([1, 1200, 1, 1]))
x422 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x426, x422)
end = time.time()
print(end-start)
