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
        self.layernorm19 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear59 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x442):
        x443=self.layernorm19(x442)
        x444=self.linear59(x443)
        return x444

m = M().eval()
x442 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x442)
end = time.time()
print(end-start)
