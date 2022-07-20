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
        self.sigmoid16 = Sigmoid()

    def forward(self, x252, x248):
        x253=self.sigmoid16(x252)
        x254=operator.mul(x253, x248)
        return x254

m = M().eval()
x252 = torch.randn(torch.Size([1, 768, 1, 1]))
x248 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x252, x248)
end = time.time()
print(end-start)
