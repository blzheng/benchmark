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
        self.linear29 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x178):
        x179=self.linear29(x178)
        x180=torch.permute(x179, [0, 3, 1, 2])
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
