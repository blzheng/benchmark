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
        self.layernorm5 = LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        self.linear10 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x70):
        x71=self.layernorm5(x70)
        x72=self.linear10(x71)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
