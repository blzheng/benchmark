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
        self.layernorm31 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear62 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu31 = GELU(approximate='none')

    def forward(self, x362):
        x363=self.layernorm31(x362)
        x364=self.linear62(x363)
        x365=self.gelu31(x364)
        return x365

m = M().eval()
x362 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x362)
end = time.time()
print(end-start)
