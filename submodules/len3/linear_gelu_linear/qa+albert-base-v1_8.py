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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x357):
        x358=self.linear5(x357)
        x359=torch._C._nn.gelu(x358)
        x360=self.linear6(x359)
        return x360

m = M().eval()
x357 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x357)
end = time.time()
print(end-start)
