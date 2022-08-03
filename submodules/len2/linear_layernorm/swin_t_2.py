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
        self.linear22 = Linear(in_features=1536, out_features=768, bias=False)
        self.layernorm24 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x256):
        x257=self.linear22(x256)
        x258=self.layernorm24(x257)
        return x258

m = M().eval()
x256 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x256)
end = time.time()
print(end-start)
