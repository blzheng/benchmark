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

    def forward(self, x521):
        x522=operator.add(x521, (256,))
        return x522

m = M().eval()
x521 = (1, 384, )
start = time.time()
output = m(x521)
end = time.time()
print(end-start)
