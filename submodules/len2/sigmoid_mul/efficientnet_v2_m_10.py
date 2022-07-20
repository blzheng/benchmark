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
        self.sigmoid10 = Sigmoid()

    def forward(self, x250, x246):
        x251=self.sigmoid10(x250)
        x252=operator.mul(x251, x246)
        return x252

m = M().eval()
x250 = torch.randn(torch.Size([1, 1056, 1, 1]))
x246 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x250, x246)
end = time.time()
print(end-start)
