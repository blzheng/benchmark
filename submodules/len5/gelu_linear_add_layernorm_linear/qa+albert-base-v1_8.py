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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x358, x357):
        x359=torch._C._nn.gelu(x358)
        x360=self.linear6(x359)
        x361=operator.add(x360, x357)
        x362=self.layernorm2(x361)
        x363=self.linear1(x362)
        return x363

m = M().eval()
x358 = torch.randn(torch.Size([1, 384, 3072]))
x357 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x358, x357)
end = time.time()
print(end-start)
