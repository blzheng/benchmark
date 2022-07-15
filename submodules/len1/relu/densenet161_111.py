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
        self.relu111 = ReLU(inplace=True)

    def forward(self, x394):
        x395=self.relu111(x394)
        return x395

m = M().eval()
x394 = torch.randn(torch.Size([1, 2112, 14, 14]))
start = time.time()
output = m(x394)
end = time.time()
print(end-start)
