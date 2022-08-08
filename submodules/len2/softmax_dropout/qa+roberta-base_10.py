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
        self.dropout31 = Dropout(p=0.1, inplace=False)

    def forward(self, x471):
        x472=torch.nn.functional.softmax(x471,dim=-1, _stacklevel=3, dtype=None)
        x473=self.dropout31(x472)
        return x473

m = M().eval()
x471 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x471)
end = time.time()
print(end-start)
