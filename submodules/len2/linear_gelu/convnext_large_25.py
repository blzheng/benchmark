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
        self.linear50 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu25 = GELU(approximate='none')

    def forward(self, x297):
        x298=self.linear50(x297)
        x299=self.gelu25(x298)
        return x299

m = M().eval()
x297 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
