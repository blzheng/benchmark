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
        self.sigmoid29 = Sigmoid()

    def forward(self, x458, x454):
        x459=self.sigmoid29(x458)
        x460=operator.mul(x459, x454)
        return x460

m = M().eval()
x458 = torch.randn(torch.Size([1, 1200, 1, 1]))
x454 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x458, x454)
end = time.time()
print(end-start)
