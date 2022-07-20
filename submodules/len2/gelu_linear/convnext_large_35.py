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
        self.gelu35 = GELU(approximate='none')
        self.linear71 = Linear(in_features=6144, out_features=1536, bias=True)

    def forward(self, x414):
        x415=self.gelu35(x414)
        x416=self.linear71(x415)
        return x416

m = M().eval()
x414 = torch.randn(torch.Size([1, 7, 7, 6144]))
start = time.time()
output = m(x414)
end = time.time()
print(end-start)
