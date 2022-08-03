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
        self.linear62 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x449):
        x451=self.linear62(x449)
        return x451

m = M().eval()
x449 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x449)
end = time.time()
print(end-start)
