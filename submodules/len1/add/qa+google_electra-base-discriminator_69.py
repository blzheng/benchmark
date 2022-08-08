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

    def forward(self, x436):
        x437=operator.add(x436, (768,))
        return x437

m = M().eval()
x436 = (1, 384, )
start = time.time()
output = m(x436)
end = time.time()
print(end-start)
