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

    def forward(self, x14):
        x15=torch.flatten(x14, 1)
        x16=self.dropout0(x15)
        return x16

m = M().eval()
x14 = torch.randn(torch.Size([1, 256, 6, 6]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
