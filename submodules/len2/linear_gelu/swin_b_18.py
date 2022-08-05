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
        self.linear38 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu18 = GELU(approximate='none')

    def forward(self, x449):
        x450=self.linear38(x449)
        x451=self.gelu18(x450)
        return x451

m = M().eval()
x449 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x449)
end = time.time()
print(end-start)
