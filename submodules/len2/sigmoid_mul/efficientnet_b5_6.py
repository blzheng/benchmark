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
        self.sigmoid6 = Sigmoid()

    def forward(self, x96, x92):
        x97=self.sigmoid6(x96)
        x98=operator.mul(x97, x92)
        return x98

m = M().eval()
x96 = torch.randn(torch.Size([1, 240, 1, 1]))
x92 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x96, x92)
end = time.time()
print(end-start)
