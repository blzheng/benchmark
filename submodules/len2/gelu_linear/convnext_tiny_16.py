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
        self.gelu16 = GELU(approximate='none')
        self.linear33 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x205):
        x206=self.gelu16(x205)
        x207=self.linear33(x206)
        return x207

m = M().eval()
x205 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
