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
        self.linear26 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.linear27 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x165):
        x166=self.linear26(x165)
        x167=self.gelu13(x166)
        x168=self.linear27(x167)
        return x168

m = M().eval()
x165 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
