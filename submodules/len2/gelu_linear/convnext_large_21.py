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
        self.linear43 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x254):
        x255=self.gelu21(x254)
        x256=self.linear43(x255)
        return x256

m = M().eval()
x254 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x254)
end = time.time()
print(end-start)
