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
        self.linear46 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout23 = Dropout(p=0.1, inplace=False)
        self.layernorm15 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear47 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear48 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout24 = Dropout(p=0.1, inplace=False)

    def forward(self, x351, x354, x323):
        x355=x351.view(x354)
        x356=self.linear46(x355)
        x357=self.dropout23(x356)
        x358=operator.add(x357, x323)
        x359=self.layernorm15(x358)
        x360=self.linear47(x359)
        x361=torch._C._nn.gelu(x360)
        x362=self.linear48(x361)
        x363=self.dropout24(x362)
        return x363

m = M().eval()
x351 = torch.randn(torch.Size([1, 384, 4, 64]))
x354 = (1, 384, 256, )
x323 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x351, x354, x323)
end = time.time()
print(end-start)
