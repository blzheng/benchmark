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

    def forward(self, x85):
        x86=operator.getitem(x85, 0)
        return x86

m = M().eval()
x85 = (1, 116, 28, 28, )
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
