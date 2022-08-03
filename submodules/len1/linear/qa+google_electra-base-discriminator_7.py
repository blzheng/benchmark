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
        self.linear7 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x70):
        x72=self.linear7(x70)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
