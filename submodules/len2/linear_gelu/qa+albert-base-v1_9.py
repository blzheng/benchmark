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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x394):
        x395=self.linear5(x394)
        x396=torch._C._nn.gelu(x395)
        return x396

m = M().eval()
x394 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x394)
end = time.time()
print(end-start)