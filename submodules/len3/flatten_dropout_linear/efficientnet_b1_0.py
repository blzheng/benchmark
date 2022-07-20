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
        self.dropout0 = Dropout(p=0.2, inplace=True)
        self.linear0 = Linear(in_features=1280, out_features=1000, bias=True)

    def forward(self, x355):
        x356=torch.flatten(x355, 1)
        x357=self.dropout0(x356)
        x358=self.linear0(x357)
        return x358

m = M().eval()
x355 = torch.randn(torch.Size([1, 1280, 1, 1]))
start = time.time()
output = m(x355)
end = time.time()
print(end-start)
