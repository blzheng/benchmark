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
        self.layernorm9 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear29 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x232):
        x233=self.layernorm9(x232)
        x234=self.linear29(x233)
        return x234

m = M().eval()
x232 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
