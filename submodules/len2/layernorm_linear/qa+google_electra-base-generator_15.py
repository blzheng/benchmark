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
        self.layernorm16 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear49 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x364):
        x365=self.layernorm16(x364)
        x366=self.linear49(x365)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
