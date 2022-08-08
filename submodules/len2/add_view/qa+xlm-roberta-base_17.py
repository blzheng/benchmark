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

    def forward(self, x206, x204):
        x207=operator.add(x206, (12, 64))
        x208=x204.view(x207)
        return x208

m = M().eval()
x206 = (1, 384, )
x204 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x206, x204)
end = time.time()
print(end-start)
