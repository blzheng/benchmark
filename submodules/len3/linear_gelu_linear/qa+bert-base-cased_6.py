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
        self.linear40 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear41 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x316):
        x317=self.linear40(x316)
        x318=torch._C._nn.gelu(x317)
        x319=self.linear41(x318)
        return x319

m = M().eval()
x316 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
