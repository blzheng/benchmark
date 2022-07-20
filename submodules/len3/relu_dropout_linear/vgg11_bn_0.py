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

    def forward(self, x32):
        x33=self.relu8(x32)
        x34=self.dropout0(x33)
        x35=self.linear1(x34)
        return x35

m = M().eval()
x32 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
