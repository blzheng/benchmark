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

    def forward(self, x508, x503):
        x509=operator.mul(x508, x503)
        return x509

m = M().eval()
x508 = torch.randn(torch.Size([1, 1344, 1, 1]))
x503 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x508, x503)
end = time.time()
print(end-start)
