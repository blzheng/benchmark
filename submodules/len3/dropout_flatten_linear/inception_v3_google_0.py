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
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear0 = Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x324):
        x325=self.dropout0(x324)
        x326=torch.flatten(x325, 1)
        x327=self.linear0(x326)
        return x327

m = M().eval()
x324 = torch.randn(torch.Size([1, 2048, 1, 1]))
start = time.time()
output = m(x324)
end = time.time()
print(end-start)
