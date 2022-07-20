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
        self.sigmoid24 = Sigmoid()

    def forward(self, x377, x373):
        x378=self.sigmoid24(x377)
        x379=operator.mul(x378, x373)
        return x379

m = M().eval()
x377 = torch.randn(torch.Size([1, 960, 1, 1]))
x373 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x377, x373)
end = time.time()
print(end-start)
