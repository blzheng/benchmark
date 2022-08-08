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

    def forward(self, x422, x408):
        x423=operator.add(x422, (4, 64))
        x424=x408.view(x423)
        return x424

m = M().eval()
x422 = (1, 384, )
x408 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x422, x408)
end = time.time()
print(end-start)
