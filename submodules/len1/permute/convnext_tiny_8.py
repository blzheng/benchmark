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

    def forward(self, ):
        x40=x39.permute(0, 2, 3, 1)
        return x40

m = M().eval()
start = time.time()
output = m()
end = time.time()
print(end-start)
