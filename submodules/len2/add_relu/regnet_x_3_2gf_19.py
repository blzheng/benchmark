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
        self.relu60 = ReLU(inplace=True)

    def forward(self, x199, x207):
        x208=operator.add(x199, x207)
        x209=self.relu60(x208)
        return x209

m = M().eval()
x199 = torch.randn(torch.Size([1, 432, 14, 14]))
x207 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x199, x207)
end = time.time()
print(end-start)
