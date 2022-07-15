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
        self.linear36 = Linear(in_features=768, out_features=1000, bias=True)

    def forward(self, x229):
        x230=self.linear36(x229)
        return x230

m = M().eval()
x229 = torch.randn(torch.Size([1, 768]))
start = time.time()
output = m(x229)
end = time.time()
print(end-start)
