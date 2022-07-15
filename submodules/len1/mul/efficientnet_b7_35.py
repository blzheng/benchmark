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

    def forward(self, x552, x547):
        x553=operator.mul(x552, x547)
        return x553

m = M().eval()
x552 = torch.randn(torch.Size([1, 1344, 1, 1]))
x547 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x552, x547)
end = time.time()
print(end-start)
