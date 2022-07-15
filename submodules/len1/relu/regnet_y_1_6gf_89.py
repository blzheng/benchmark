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
        self.relu89 = ReLU(inplace=True)

    def forward(self, x363):
        x364=self.relu89(x363)
        return x364

m = M().eval()
x363 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x363)
end = time.time()
print(end-start)
