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
        self.linear47 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear48 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout24 = Dropout(p=0.1, inplace=False)
        self.layernorm16 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear49 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x359, x359):
        x360=self.linear47(x359)
        x361=torch._C._nn.gelu(x360)
        x362=self.linear48(x361)
        x363=self.dropout24(x362)
        x364=operator.add(x363, x359)
        x365=self.layernorm16(x364)
        x366=self.linear49(x365)
        return x366

m = M().eval()
x359 = torch.randn(torch.Size([1, 384, 256]))
x359 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x359, x359)
end = time.time()
print(end-start)
