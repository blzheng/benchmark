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
        self.gelu12 = GELU(approximate='none')
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x312):
        x313=self.gelu12(x312)
        x314=self.dropout24(x313)
        x315=self.linear27(x314)
        return x315

m = M().eval()
x312 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
