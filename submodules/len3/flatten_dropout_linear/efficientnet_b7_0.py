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
        self.dropout0 = Dropout(p=0.5, inplace=True)
        self.linear0 = Linear(in_features=2560, out_features=1000, bias=True)

    def forward(self, x861):
        x862=torch.flatten(x861, 1)
        x863=self.dropout0(x862)
        x864=self.linear0(x863)
        return x864

m = M().eval()
x861 = torch.randn(torch.Size([1, 2560, 1, 1]))
start = time.time()
output = m(x861)
end = time.time()
print(end-start)
