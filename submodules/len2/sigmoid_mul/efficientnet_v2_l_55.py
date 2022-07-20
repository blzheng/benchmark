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
        self.sigmoid55 = Sigmoid()

    def forward(self, x999, x995):
        x1000=self.sigmoid55(x999)
        x1001=operator.mul(x1000, x995)
        return x1001

m = M().eval()
x999 = torch.randn(torch.Size([1, 3840, 1, 1]))
x995 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x999, x995)
end = time.time()
print(end-start)
