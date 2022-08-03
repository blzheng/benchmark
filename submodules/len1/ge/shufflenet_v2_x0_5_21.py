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
        x89=operator.getitem(x85, 3)
        return x89

m = M().eval()
x85 = (1, 48, 28, 28, )
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
