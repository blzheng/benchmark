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

    def forward(self, x199, x202):
        x203=x199.view(x202)
        return x203

m = M().eval()
x199 = torch.randn(torch.Size([1, 384, 256]))
x202 = (1, 384, 4, 64, )
start = time.time()
output = m(x199, x202)
end = time.time()
print(end-start)
