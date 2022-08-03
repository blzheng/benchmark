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
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)

    def forward(self, x21):
        x22=self.dropout0(x21)
        x23=self.linear1(x22)
        return x23

m = M().eval()
x21 = torch.randn(torch.Size([1, 56, 56, 384]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
