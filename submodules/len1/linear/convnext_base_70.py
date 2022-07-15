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
        self.linear70 = Linear(in_features=1024, out_features=4096, bias=True)

    def forward(self, x413):
        x414=self.linear70(x413)
        return x414

m = M().eval()
x413 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x413)
end = time.time()
print(end-start)
