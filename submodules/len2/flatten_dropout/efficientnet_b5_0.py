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
        self.dropout0 = Dropout(p=0.4, inplace=True)

    def forward(self, x608):
        x609=torch.flatten(x608, 1)
        x610=self.dropout0(x609)
        return x610

m = M().eval()
x608 = torch.randn(torch.Size([1, 2048, 1, 1]))
start = time.time()
output = m(x608)
end = time.time()
print(end-start)
