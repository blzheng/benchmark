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
        self.linear47 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu22 = GELU(approximate='none')

    def forward(self, x549):
        x550=self.linear47(x549)
        x551=self.gelu22(x550)
        return x551

m = M().eval()
x549 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x549)
end = time.time()
print(end-start)