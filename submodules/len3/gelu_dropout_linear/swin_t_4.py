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
        self.gelu4 = GELU(approximate=none)
        self.dropout8 = Dropout(p=0.0, inplace=False)
        self.linear11 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x128):
        x129=self.gelu4(x128)
        x130=self.dropout8(x129)
        x131=self.linear11(x130)
        return x131

m = M().eval()
x128 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
