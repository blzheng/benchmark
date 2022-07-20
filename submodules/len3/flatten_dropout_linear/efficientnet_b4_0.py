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
        self.dropout0 = Dropout(p=0.4, inplace=True)
        self.linear0 = Linear(in_features=1792, out_features=1000, bias=True)

    def forward(self, x499):
        x500=torch.flatten(x499, 1)
        x501=self.dropout0(x500)
        x502=self.linear0(x501)
        return x502

m = M().eval()
x499 = torch.randn(torch.Size([1, 1792, 1, 1]))
start = time.time()
output = m(x499)
end = time.time()
print(end-start)
