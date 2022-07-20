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
        self.linear28 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.linear29 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x176):
        x177=self.linear28(x176)
        x178=self.gelu14(x177)
        x179=self.linear29(x178)
        return x179

m = M().eval()
x176 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
