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

    def forward(self, x131):
        x132=operator.getitem(x131, 0)
        return x132

m = M().eval()
x131 = (1, 488, 14, 14, )
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
