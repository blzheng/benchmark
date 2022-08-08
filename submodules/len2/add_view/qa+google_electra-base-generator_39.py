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

    def forward(self, x437, x435):
        x438=operator.add(x437, (256,))
        x439=x435.view(x438)
        return x439

m = M().eval()
x437 = (1, 384, )
x435 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x437, x435)
end = time.time()
print(end-start)
