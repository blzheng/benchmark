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
        self.dropout0 = Dropout(p=0.5, inplace=False)

    def forward(self, x367):
        x368=self.dropout0(x367)
        return x368

m = M().eval()
x367 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x367)
end = time.time()
print(end-start)
