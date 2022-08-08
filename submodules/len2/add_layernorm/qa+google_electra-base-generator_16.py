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

    def forward(self, x363, x359):
        x364=operator.add(x363, x359)
        x365=self.layernorm16(x364)
        return x365

m = M().eval()
x363 = torch.randn(torch.Size([1, 384, 256]))
x359 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x363, x359)
end = time.time()
print(end-start)
