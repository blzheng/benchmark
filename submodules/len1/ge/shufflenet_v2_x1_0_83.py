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

    def forward(self, x331):
        x335=operator.getitem(x331, 3)
        return x335

m = M().eval()
x331 = (1, 464, 7, 7, )
start = time.time()
output = m(x331)
end = time.time()
print(end-start)
