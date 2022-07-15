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

    def forward(self, x492, x477):
        x493=operator.add(x492, x477)
        return x493

m = M().eval()
x492 = torch.randn(torch.Size([1, 224, 14, 14]))
x477 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x492, x477)
end = time.time()
print(end-start)
