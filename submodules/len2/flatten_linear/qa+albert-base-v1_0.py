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

    def forward(self, x56):
        x57=x56.flatten(2)
        x58=self.linear4(x57)
        return x58

m = M().eval()
x56 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
