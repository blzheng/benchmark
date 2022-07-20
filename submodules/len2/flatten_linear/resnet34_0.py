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
        self.linear0 = Linear(in_features=512, out_features=1000, bias=True)

    def forward(self, x123):
        x124=torch.flatten(x123, 1)
        x125=self.linear0(x124)
        return x125

m = M().eval()
x123 = torch.randn(torch.Size([1, 512, 1, 1]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)
