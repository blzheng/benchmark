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
        self.dropout28 = Dropout(p=0.1, inplace=False)

    def forward(self, x430):
        x431=torch.nn.functional.softmax(x430,dim=-1, _stacklevel=3, dtype=None)
        x432=self.dropout28(x431)
        return x432

m = M().eval()
x430 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x430)
end = time.time()
print(end-start)
