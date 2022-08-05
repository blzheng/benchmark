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
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x486, x484):
        x487=self.linear65(x486)
        x488=self.dropout33(x487)
        x489=operator.add(x488, x484)
        return x489

m = M().eval()
x486 = torch.randn(torch.Size([1, 384, 3072]))
x484 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x486, x484)
end = time.time()
print(end-start)
