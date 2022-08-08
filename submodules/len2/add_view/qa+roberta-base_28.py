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

    def forward(self, x326, x324):
        x327=operator.add(x326, (12, 64))
        x328=x324.view(x327)
        return x328

m = M().eval()
x326 = (1, 384, )
x324 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x326, x324)
end = time.time()
print(end-start)
