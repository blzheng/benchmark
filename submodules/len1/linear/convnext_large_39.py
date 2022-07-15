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
        self.linear39 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x233):
        x234=self.linear39(x233)
        return x234

m = M().eval()
x233 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x233)
end = time.time()
print(end-start)
