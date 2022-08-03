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

    def forward(self, x153):
        x157=operator.getitem(x153, 3)
        return x157

m = M().eval()
x153 = (1, 352, 14, 14, )
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
