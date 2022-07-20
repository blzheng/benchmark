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
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x41):
        x42=self.dropout0(x41)
        x43=self.linear1(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
