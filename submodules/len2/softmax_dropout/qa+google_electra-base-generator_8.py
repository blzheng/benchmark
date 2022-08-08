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
        self.dropout25 = Dropout(p=0.1, inplace=False)

    def forward(self, x388):
        x389=torch.nn.functional.softmax(x388,dim=-1, _stacklevel=3, dtype=None)
        x390=self.dropout25(x389)
        return x390

m = M().eval()
x388 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)
