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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x238):
        x239=self.dropout1(x238)
        return x239

m = M().eval()
x238 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x238)
end = time.time()
print(end-start)
