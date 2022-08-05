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
        self.layernorm38 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x410, x424):
        x425=operator.add(x410, x424)
        x426=self.layernorm38(x425)
        x427=self.linear36(x426)
        return x427

m = M().eval()
x410 = torch.randn(torch.Size([1, 14, 14, 384]))
x424 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x410, x424)
end = time.time()
print(end-start)
