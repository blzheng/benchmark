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
        self.dropout34 = Dropout(p=0.1, inplace=False)

    def forward(self, x514):
        x515=self.dropout34(x514)
        return x515

m = M().eval()
x514 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x514)
end = time.time()
print(end-start)
