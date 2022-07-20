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
        self.linear69 = Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, x404):
        x405=self.linear69(x404)
        x406=torch.permute(x405, [0, 3, 1, 2])
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 7, 7, 4096]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
