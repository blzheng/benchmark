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
        self.linear24 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu12 = GELU(approximate='none')

    def forward(self, x154):
        x155=self.linear24(x154)
        x156=self.gelu12(x155)
        return x156

m = M().eval()
x154 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x154)
end = time.time()
print(end-start)
