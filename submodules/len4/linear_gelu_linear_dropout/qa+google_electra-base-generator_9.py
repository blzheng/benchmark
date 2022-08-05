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
        self.linear59 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear60 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout30 = Dropout(p=0.1, inplace=False)

    def forward(self, x443):
        x444=self.linear59(x443)
        x445=torch._C._nn.gelu(x444)
        x446=self.linear60(x445)
        x447=self.dropout30(x446)
        return x447

m = M().eval()
x443 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x443)
end = time.time()
print(end-start)
