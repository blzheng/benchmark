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

    def forward(self, x427, x423):
        x428=self.sigmoid19(x427)
        x429=operator.mul(x428, x423)
        return x429

m = M().eval()
x427 = torch.randn(torch.Size([1, 1344, 1, 1]))
x423 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x427, x423)
end = time.time()
print(end-start)
