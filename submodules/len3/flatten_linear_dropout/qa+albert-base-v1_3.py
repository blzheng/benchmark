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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x167):
        x168=x167.flatten(2)
        x169=self.linear4(x168)
        x170=self.dropout2(x169)
        return x170

m = M().eval()
x167 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)
