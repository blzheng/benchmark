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
        self.linear65 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x486):
        x487=self.linear65(x486)
        return x487

m = M().eval()
x486 = torch.randn(torch.Size([1, 384, 3072]))
start = time.time()
output = m(x486)
end = time.time()
print(end-start)
