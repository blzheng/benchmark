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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x348, x343):
        x349=torch.nn.functional.softmax(x348,dim=-1, _stacklevel=3, dtype=None)
        x350=self.dropout1(x349)
        x351=torch.matmul(x350, x343)
        return x351

m = M().eval()
x348 = torch.randn(torch.Size([1, 12, 384, 384]))
x343 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x348, x343)
end = time.time()
print(end-start)
