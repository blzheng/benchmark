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
        self.layernorm34 = LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
        self.linear68 = Linear(in_features=1536, out_features=6144, bias=True)
        self.gelu34 = GELU(approximate='none')
        self.linear69 = Linear(in_features=6144, out_features=1536, bias=True)

    def forward(self, x401):
        x402=self.layernorm34(x401)
        x403=self.linear68(x402)
        x404=self.gelu34(x403)
        x405=self.linear69(x404)
        return x405

m = M().eval()
x401 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x401)
end = time.time()
print(end-start)
