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
        self.linear2 = Linear(in_features=192, out_features=768, bias=True)

    def forward(self, x21):
        x22=self.linear2(x21)
        return x22

m = M().eval()
x21 = torch.randn(torch.Size([1, 56, 56, 192]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
