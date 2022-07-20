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
        self.sigmoid12 = Sigmoid()

    def forward(self, x315, x311):
        x316=self.sigmoid12(x315)
        x317=operator.mul(x316, x311)
        return x317

m = M().eval()
x315 = torch.randn(torch.Size([1, 1344, 1, 1]))
x311 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x315, x311)
end = time.time()
print(end-start)
