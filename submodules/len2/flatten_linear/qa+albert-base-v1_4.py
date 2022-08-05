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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x204):
        x205=x204.flatten(2)
        x206=self.linear4(x205)
        return x206

m = M().eval()
x204 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)