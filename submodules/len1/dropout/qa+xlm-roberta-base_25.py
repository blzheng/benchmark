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
        self.dropout25 = Dropout(p=0.1, inplace=False)

    def forward(self, x388):
        x389=self.dropout25(x388)
        return x389

m = M().eval()
x388 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)
