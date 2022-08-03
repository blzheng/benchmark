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

    def forward(self, x241):
        x244=operator.getitem(x241, 2)
        return x244

m = M().eval()
x241 = (1, 352, 14, 14, )
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
