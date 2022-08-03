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
        self.dropout28 = Dropout(p=0.0, inplace=False)
        self.linear31 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x359):
        x360=self.dropout28(x359)
        x361=self.linear31(x360)
        return x361

m = M().eval()
x359 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x359)
end = time.time()
print(end-start)
