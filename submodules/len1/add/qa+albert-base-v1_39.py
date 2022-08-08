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

    def forward(self, x261):
        x262=operator.add(x261, (12, 64))
        return x262

m = M().eval()
x261 = (1, 384, )
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
