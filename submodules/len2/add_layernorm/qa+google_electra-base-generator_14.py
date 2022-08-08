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
        self.layernorm14 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x321, x317):
        x322=operator.add(x321, x317)
        x323=self.layernorm14(x322)
        return x323

m = M().eval()
x321 = torch.randn(torch.Size([1, 384, 256]))
x317 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x321, x317)
end = time.time()
print(end-start)
