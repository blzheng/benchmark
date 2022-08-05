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
        self.linear10 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.1, inplace=False)

    def forward(self, x99, x102, x71):
        x103=x99.view(x102)
        x104=self.linear10(x103)
        x105=self.dropout5(x104)
        x106=operator.add(x105, x71)
        return x106

m = M().eval()
x99 = torch.randn(torch.Size([1, 384, 4, 64]))
x102 = (1, 384, 256, )
x71 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x99, x102, x71)
end = time.time()
print(end-start)
