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
        self.linear46 = Linear(in_features=2048, out_features=1024, bias=False)

    def forward(self, x532):
        x533=self.linear46(x532)
        return x533

m = M().eval()
x532 = torch.randn(torch.Size([1, 7, 7, 2048]))
start = time.time()
output = m(x532)
end = time.time()
print(end-start)
