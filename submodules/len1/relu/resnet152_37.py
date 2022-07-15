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
        self.relu37 = ReLU(inplace=True)

    def forward(self, x132):
        x133=self.relu37(x132)
        return x133

m = M().eval()
x132 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x132)
end = time.time()
print(end-start)
