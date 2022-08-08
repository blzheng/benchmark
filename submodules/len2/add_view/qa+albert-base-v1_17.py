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

    def forward(self, x229, x217):
        x230=operator.add(x229, (12, 64))
        x231=x217.view(x230)
        return x231

m = M().eval()
x229 = (1, 384, )
x217 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x229, x217)
end = time.time()
print(end-start)
