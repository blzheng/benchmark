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

    def forward(self, x155, x143):
        x156=operator.add(x155, (12, 64))
        x157=x143.view(x156)
        return x157

m = M().eval()
x155 = (1, 384, )
x143 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x155, x143)
end = time.time()
print(end-start)
