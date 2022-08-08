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

    def forward(self, x169, x155):
        x170=operator.add(x169, (12, 64))
        x171=x155.view(x170)
        return x171

m = M().eval()
x169 = (1, 384, )
x155 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x169, x155)
end = time.time()
print(end-start)
