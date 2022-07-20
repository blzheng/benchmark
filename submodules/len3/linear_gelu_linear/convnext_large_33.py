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
        self.linear66 = Linear(in_features=1536, out_features=6144, bias=True)
        self.gelu33 = GELU(approximate='none')
        self.linear67 = Linear(in_features=6144, out_features=1536, bias=True)

    def forward(self, x391):
        x392=self.linear66(x391)
        x393=self.gelu33(x392)
        x394=self.linear67(x393)
        return x394

m = M().eval()
x391 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x391)
end = time.time()
print(end-start)
