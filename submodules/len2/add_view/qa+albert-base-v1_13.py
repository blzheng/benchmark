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

    def forward(self, x187, x179):
        x188=operator.add(x187, (12, 64))
        x189=x179.view(x188)
        return x189

m = M().eval()
x187 = (1, 384, )
x179 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x187, x179)
end = time.time()
print(end-start)
