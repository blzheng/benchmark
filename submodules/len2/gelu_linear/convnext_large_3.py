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
        self.gelu3 = GELU(approximate='none')
        self.linear7 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x50):
        x51=self.gelu3(x50)
        x52=self.linear7(x51)
        return x52

m = M().eval()
x50 = torch.randn(torch.Size([1, 28, 28, 1536]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
