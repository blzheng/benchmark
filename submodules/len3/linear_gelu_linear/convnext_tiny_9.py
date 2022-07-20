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
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x121):
        x122=self.linear18(x121)
        x123=self.gelu9(x122)
        x124=self.linear19(x123)
        return x124

m = M().eval()
x121 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
