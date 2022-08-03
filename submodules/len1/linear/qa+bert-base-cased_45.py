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
        self.linear45 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x354):
        x355=self.linear45(x354)
        return x355

m = M().eval()
x354 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x354)
end = time.time()
print(end-start)
