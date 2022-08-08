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

    def forward(self, x338, x324):
        x339=operator.add(x338, (4, 64))
        x340=x324.view(x339)
        return x340

m = M().eval()
x338 = (1, 384, )
x324 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x338, x324)
end = time.time()
print(end-start)
