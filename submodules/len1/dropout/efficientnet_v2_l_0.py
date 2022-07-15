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
        self.dropout0 = Dropout(p=0.4, inplace=True)

    def forward(self, x1090):
        x1091=self.dropout0(x1090)
        return x1091

m = M().eval()
x1090 = torch.randn(torch.Size([1, 1280]))
start = time.time()
output = m(x1090)
end = time.time()
print(end-start)
