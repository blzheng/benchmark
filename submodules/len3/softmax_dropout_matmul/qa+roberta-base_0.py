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

    def forward(self, x51, x41):
        x52=torch.nn.functional.softmax(x51,dim=-1, _stacklevel=3, dtype=None)
        x53=self.dropout1(x52)
        x54=torch.matmul(x53, x41)
        return x54

m = M().eval()
x51 = torch.randn(torch.Size([1, 12, 384, 384]))
x41 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x51, x41)
end = time.time()
print(end-start)
