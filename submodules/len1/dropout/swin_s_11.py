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
        self.dropout11 = Dropout(p=0.0, inplace=False)

    def forward(self, x154):
        x155=self.dropout11(x154)
        return x155

m = M().eval()
x154 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x154)
end = time.time()
print(end-start)
