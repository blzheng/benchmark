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

    def forward(self, x200, x195):
        x201=torch.nn.functional.softmax(x200,dim=-1, _stacklevel=3, dtype=None)
        x202=self.dropout1(x201)
        x203=torch.matmul(x202, x195)
        return x203

m = M().eval()
x200 = torch.randn(torch.Size([1, 12, 384, 384]))
x195 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x200, x195)
end = time.time()
print(end-start)
