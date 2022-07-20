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
        self.linear0 = Linear(in_features=9216, out_features=4096, bias=True)

    def forward(self, x15):
        x16=self.dropout0(x15)
        x17=self.linear0(x16)
        return x17

m = M().eval()
x15 = torch.randn(torch.Size([1, 9216]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
