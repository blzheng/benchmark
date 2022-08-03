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

    def forward(self, x353):
        x355=operator.getitem(x353, 1)
        return x355

m = M().eval()
x353 = (1, 704, 7, 7, )
start = time.time()
output = m(x353)
end = time.time()
print(end-start)
