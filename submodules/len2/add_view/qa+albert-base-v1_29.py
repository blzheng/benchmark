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

    def forward(self, x377, x365):
        x378=operator.add(x377, (12, 64))
        x379=x365.view(x378)
        return x379

m = M().eval()
x377 = (1, 384, )
x365 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x377, x365)
end = time.time()
print(end-start)
