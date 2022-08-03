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

    def forward(self, x0):
        x9=operator.getitem(x0, (slice(None, None, None), 2))
        return x9

m = M().eval()
x0 = torch.randn(torch.Size([1, 3, 224, 224]))
start = time.time()
output = m(x0)
end = time.time()
print(end-start)
