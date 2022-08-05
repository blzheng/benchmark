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
        self.linear29 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout27 = Dropout(p=0.0, inplace=False)

    def forward(self, x337):
        x338=self.linear29(x337)
        x339=self.dropout27(x338)
        return x339

m = M().eval()
x337 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x337)
end = time.time()
print(end-start)
