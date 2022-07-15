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
        self.relu47 = ReLU(inplace=True)

    def forward(self, x164):
        x165=self.relu47(x164)
        return x165

m = M().eval()
x164 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
