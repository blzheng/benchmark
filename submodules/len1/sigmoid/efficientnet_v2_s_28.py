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
        self.sigmoid28 = Sigmoid()

    def forward(self, x517):
        x518=self.sigmoid28(x517)
        return x518

m = M().eval()
x517 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x517)
end = time.time()
print(end-start)
