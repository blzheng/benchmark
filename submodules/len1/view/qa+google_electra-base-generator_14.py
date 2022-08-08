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

    def forward(self, x156, x171):
        x172=x156.view(x171)
        return x172

m = M().eval()
x156 = torch.randn(torch.Size([1, 384, 256]))
x171 = (1, 384, 4, 64, )
start = time.time()
output = m(x156, x171)
end = time.time()
print(end-start)
