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
        self.dropout0 = Dropout(p=0.2, inplace=False)

    def forward(self, x207):
        x208=torch.flatten(x207, 1)
        x209=self.dropout0(x208)
        return x209

m = M().eval()
x207 = torch.randn(torch.Size([1, 1024, 1, 1]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
