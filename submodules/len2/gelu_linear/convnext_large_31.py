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
        self.gelu31 = GELU(approximate='none')
        self.linear63 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x364):
        x365=self.gelu31(x364)
        x366=self.linear63(x365)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
