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
        self.linear36 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu18 = GELU(approximate='none')
        self.linear37 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x220):
        x221=self.linear36(x220)
        x222=self.gelu18(x221)
        x223=self.linear37(x222)
        return x223

m = M().eval()
x220 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
