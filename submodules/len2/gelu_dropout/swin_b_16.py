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
        self.gelu16 = GELU(approximate='none')
        self.dropout32 = Dropout(p=0.0, inplace=False)

    def forward(self, x404):
        x405=self.gelu16(x404)
        x406=self.dropout32(x405)
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
