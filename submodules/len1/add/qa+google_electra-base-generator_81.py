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

    def forward(self, x506):
        x507=operator.add(x506, (4, 64))
        return x507

m = M().eval()
x506 = (1, 384, )
start = time.time()
output = m(x506)
end = time.time()
print(end-start)
