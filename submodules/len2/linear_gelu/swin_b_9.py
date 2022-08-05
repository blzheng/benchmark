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
        self.linear20 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu9 = GELU(approximate='none')

    def forward(self, x242):
        x243=self.linear20(x242)
        x244=self.gelu9(x243)
        return x244

m = M().eval()
x242 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x242)
end = time.time()
print(end-start)
