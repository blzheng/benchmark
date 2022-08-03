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
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x274):
        x275=self.linear34(x274)
        return x275

m = M().eval()
x274 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
