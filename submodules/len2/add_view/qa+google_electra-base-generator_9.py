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

    def forward(self, x123, x121):
        x124=operator.add(x123, (4, 64))
        x125=x121.view(x124)
        return x125

m = M().eval()
x123 = (1, 384, )
x121 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x123, x121)
end = time.time()
print(end-start)
