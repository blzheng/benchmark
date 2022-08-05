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
        self.dropout36 = Dropout(p=0.0, inplace=False)
        self.linear39 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout37 = Dropout(p=0.0, inplace=False)

    def forward(self, x451):
        x452=self.dropout36(x451)
        x453=self.linear39(x452)
        x454=self.dropout37(x453)
        return x454

m = M().eval()
x451 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x451)
end = time.time()
print(end-start)
