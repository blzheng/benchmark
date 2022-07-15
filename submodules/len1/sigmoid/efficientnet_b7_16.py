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

    def forward(self, x251):
        x252=self.sigmoid16(x251)
        return x252

m = M().eval()
x251 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
