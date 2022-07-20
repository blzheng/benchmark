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
        self.linear19 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x123):
        x124=self.linear19(x123)
        x125=torch.permute(x124, [0, 3, 1, 2])
        return x125

m = M().eval()
x123 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)
