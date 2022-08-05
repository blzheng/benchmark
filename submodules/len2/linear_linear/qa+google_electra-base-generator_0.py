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
        self.linear0 = Linear(in_features=768, out_features=256, bias=True)
        self.linear1 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x28):
        x29=self.linear0(x28)
        x30=self.linear1(x29)
        return x30

m = M().eval()
x28 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
