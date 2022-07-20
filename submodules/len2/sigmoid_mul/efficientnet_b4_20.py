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
        self.sigmoid20 = Sigmoid()

    def forward(self, x317, x313):
        x318=self.sigmoid20(x317)
        x319=operator.mul(x318, x313)
        return x319

m = M().eval()
x317 = torch.randn(torch.Size([1, 960, 1, 1]))
x313 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x317, x313)
end = time.time()
print(end-start)
