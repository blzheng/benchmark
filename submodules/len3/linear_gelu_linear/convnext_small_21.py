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
        self.linear42 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu21 = GELU(approximate='none')
        self.linear43 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x253):
        x254=self.linear42(x253)
        x255=self.gelu21(x254)
        x256=self.linear43(x255)
        return x256

m = M().eval()
x253 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
