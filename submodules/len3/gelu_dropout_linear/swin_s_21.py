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
        self.gelu21 = GELU(approximate='none')
        self.dropout42 = Dropout(p=0.0, inplace=False)
        self.linear45 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x519):
        x520=self.gelu21(x519)
        x521=self.dropout42(x520)
        x522=self.linear45(x521)
        return x522

m = M().eval()
x519 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x519)
end = time.time()
print(end-start)
