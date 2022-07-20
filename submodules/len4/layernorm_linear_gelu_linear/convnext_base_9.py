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
        self.layernorm9 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear18 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.linear19 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x120):
        x121=self.layernorm9(x120)
        x122=self.linear18(x121)
        x123=self.gelu9(x122)
        x124=self.linear19(x123)
        return x124

m = M().eval()
x120 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
