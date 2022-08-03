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
        self.linear21 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)

    def forward(self, x245):
        x246=self.linear21(x245)
        x247=self.dropout19(x246)
        return x247

m = M().eval()
x245 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x245)
end = time.time()
print(end-start)
