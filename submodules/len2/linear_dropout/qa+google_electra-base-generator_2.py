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

    def forward(self, x103):
        x104=self.linear10(x103)
        x105=self.dropout5(x104)
        return x105

m = M().eval()
x103 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
