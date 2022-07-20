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
        self.linear9 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x62):
        x63=self.linear9(x62)
        x64=torch.permute(x63, [0, 3, 1, 2])
        return x64

m = M().eval()
x62 = torch.randn(torch.Size([1, 28, 28, 1024]))
start = time.time()
output = m(x62)
end = time.time()
print(end-start)
