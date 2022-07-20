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
        self.linear32 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.linear33 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x204):
        x205=self.linear32(x204)
        x206=self.gelu16(x205)
        x207=self.linear33(x206)
        return x207

m = M().eval()
x204 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)
