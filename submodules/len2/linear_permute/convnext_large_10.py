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
        self.linear21 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x134):
        x135=self.linear21(x134)
        x136=torch.permute(x135, [0, 3, 1, 2])
        return x136

m = M().eval()
x134 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x134)
end = time.time()
print(end-start)
