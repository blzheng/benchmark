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
        self.sigmoid24 = Sigmoid()

    def forward(self, x453):
        x454=self.sigmoid24(x453)
        return x454

m = M().eval()
x453 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x453)
end = time.time()
print(end-start)
