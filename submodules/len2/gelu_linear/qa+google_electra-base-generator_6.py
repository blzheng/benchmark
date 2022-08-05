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
        self.linear42 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x318):
        x319=torch._C._nn.gelu(x318)
        x320=self.linear42(x319)
        return x320

m = M().eval()
x318 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x318)
end = time.time()
print(end-start)
