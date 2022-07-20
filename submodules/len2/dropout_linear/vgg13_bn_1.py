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
        self.dropout1 = Dropout(p=0.5, inplace=False)
        self.linear2 = Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x42):
        x43=self.dropout1(x42)
        x44=self.linear2(x43)
        return x44

m = M().eval()
x42 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
