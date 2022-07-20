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

    def forward(self, x27, x17, x38):
        x28=operator.add(x27, x17)
        x39=operator.add(x38, x28)
        return x39

m = M().eval()
x27 = torch.randn(torch.Size([1, 128, 56, 56]))
x17 = torch.randn(torch.Size([1, 128, 56, 56]))
x38 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x27, x17, x38)
end = time.time()
print(end-start)