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
        self.sigmoid54 = Sigmoid()

    def forward(self, x985):
        x986=self.sigmoid54(x985)
        return x986

m = M().eval()
x985 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x985)
end = time.time()
print(end-start)
