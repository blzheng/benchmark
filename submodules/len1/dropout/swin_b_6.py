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
        self.dropout6 = Dropout(p=0.0, inplace=False)

    def forward(self, x98):
        x99=self.dropout6(x98)
        return x99

m = M().eval()
x98 = torch.randn(torch.Size([1, 28, 28, 1024]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)