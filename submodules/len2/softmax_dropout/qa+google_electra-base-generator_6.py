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
        self.dropout19 = Dropout(p=0.1, inplace=False)

    def forward(self, x304):
        x305=torch.nn.functional.softmax(x304,dim=-1, _stacklevel=3, dtype=None)
        x306=self.dropout19(x305)
        return x306

m = M().eval()
x304 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x304)
end = time.time()
print(end-start)
