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
        self.linear58 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout29 = Dropout(p=0.1, inplace=False)
        self.layernorm19 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear59 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x439, x407):
        x440=self.linear58(x439)
        x441=self.dropout29(x440)
        x442=operator.add(x441, x407)
        x443=self.layernorm19(x442)
        x444=self.linear59(x443)
        x445=torch._C._nn.gelu(x444)
        return x445

m = M().eval()
x439 = torch.randn(torch.Size([1, 384, 256]))
x407 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x439, x407)
end = time.time()
print(end-start)
