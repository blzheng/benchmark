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
        self.sigmoid33 = Sigmoid()

    def forward(self, x519, x515):
        x520=self.sigmoid33(x519)
        x521=operator.mul(x520, x515)
        return x521

m = M().eval()
x519 = torch.randn(torch.Size([1, 1344, 1, 1]))
x515 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x519, x515)
end = time.time()
print(end-start)
