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
        self.dropout13 = Dropout(p=0.1, inplace=False)

    def forward(self, x219):
        x220=torch.nn.functional.softmax(x219,dim=-1, _stacklevel=3, dtype=None)
        x221=self.dropout13(x220)
        return x221

m = M().eval()
x219 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x219)
end = time.time()
print(end-start)
