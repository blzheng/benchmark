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
        self.dropout10 = Dropout(p=0.1, inplace=False)

    def forward(self, x178):
        x179=torch.nn.functional.softmax(x178,dim=-1, _stacklevel=3, dtype=None)
        x180=self.dropout10(x179)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
