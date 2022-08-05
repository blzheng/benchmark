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
        self.linear45 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout43 = Dropout(p=0.0, inplace=False)

    def forward(self, x521):
        x522=self.linear45(x521)
        x523=self.dropout43(x522)
        return x523

m = M().eval()
x521 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x521)
end = time.time()
print(end-start)
