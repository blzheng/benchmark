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
        self.linear0 = Linear(in_features=1792, out_features=1000, bias=True)

    def forward(self, x477):
        x478=self.linear0(x477)
        return x478

m = M().eval()
x477 = torch.randn(torch.Size([1, 1792]))
start = time.time()
output = m(x477)
end = time.time()
print(end-start)