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
        self.linear66 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x486):
        x487=torch._C._nn.gelu(x486)
        x488=self.linear66(x487)
        return x488

m = M().eval()
x486 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x486)
end = time.time()
print(end-start)
