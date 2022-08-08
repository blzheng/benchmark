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

    def forward(self, x177):
        x178=torch.nn.functional.softmax(x177,dim=-1, _stacklevel=3, dtype=None)
        x179=self.dropout10(x178)
        return x179

m = M().eval()
x177 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
