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
        self.dropout22 = Dropout(p=0.1, inplace=False)

    def forward(self, x346, x336):
        x347=torch.nn.functional.softmax(x346,dim=-1, _stacklevel=3, dtype=None)
        x348=self.dropout22(x347)
        x349=torch.matmul(x348, x336)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 4, 384, 384]))
x336 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x346, x336)
end = time.time()
print(end-start)
