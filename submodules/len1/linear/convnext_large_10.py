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
        self.linear10 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x71):
        x72=self.linear10(x71)
        return x72

m = M().eval()
x71 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
