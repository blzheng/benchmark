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
        self.dropout0 = Dropout(p=0.3, inplace=True)
        self.linear0 = Linear(in_features=1536, out_features=1000, bias=True)

    def forward(self, x404):
        x405=self.dropout0(x404)
        x406=self.linear0(x405)
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 1536]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
