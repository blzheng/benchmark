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

    def forward(self, x520, x518):
        x521=operator.add(x520, (768,))
        x522=x518.view(x521)
        return x522

m = M().eval()
x520 = (1, 384, )
x518 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x520, x518)
end = time.time()
print(end-start)
