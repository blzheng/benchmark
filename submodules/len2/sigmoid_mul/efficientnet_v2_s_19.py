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
        self.sigmoid19 = Sigmoid()

    def forward(self, x373, x369):
        x374=self.sigmoid19(x373)
        x375=operator.mul(x374, x369)
        return x375

m = M().eval()
x373 = torch.randn(torch.Size([1, 1536, 1, 1]))
x369 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x373, x369)
end = time.time()
print(end-start)
