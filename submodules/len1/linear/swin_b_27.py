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
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x314):
        x315=self.linear27(x314)
        return x315

m = M().eval()
x314 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x314)
end = time.time()
print(end-start)
