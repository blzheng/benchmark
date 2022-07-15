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

    def forward(self, x544):
        x545=self.dropout0(x544)
        return x545

m = M().eval()
x544 = torch.randn(torch.Size([1, 1280]))
start = time.time()
output = m(x544)
end = time.time()
print(end-start)
