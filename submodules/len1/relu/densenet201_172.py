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
        self.relu172 = ReLU(inplace=True)

    def forward(self, x609):
        x610=self.relu172(x609)
        return x610

m = M().eval()
x609 = torch.randn(torch.Size([1, 1472, 7, 7]))
start = time.time()
output = m(x609)
end = time.time()
print(end-start)
