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
        self.sigmoid21 = Sigmoid()

    def forward(self, x459, x455):
        x460=self.sigmoid21(x459)
        x461=operator.mul(x460, x455)
        return x461

m = M().eval()
x459 = torch.randn(torch.Size([1, 1344, 1, 1]))
x455 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x459, x455)
end = time.time()
print(end-start)
