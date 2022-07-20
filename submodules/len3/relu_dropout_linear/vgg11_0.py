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
        self.relu8 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x24):
        x25=self.relu8(x24)
        x26=self.dropout0(x25)
        x27=self.linear1(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
