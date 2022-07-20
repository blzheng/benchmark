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
        self.sigmoid3 = Sigmoid()

    def forward(self, x50, x46):
        x51=self.sigmoid3(x50)
        x52=operator.mul(x51, x46)
        return x52

m = M().eval()
x50 = torch.randn(torch.Size([1, 192, 1, 1]))
x46 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x50, x46)
end = time.time()
print(end-start)
