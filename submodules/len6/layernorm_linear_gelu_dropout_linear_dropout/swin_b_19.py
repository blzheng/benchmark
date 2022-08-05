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
        self.layernorm42 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu19 = GELU(approximate='none')
        self.dropout38 = Dropout(p=0.0, inplace=False)
        self.linear41 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout39 = Dropout(p=0.0, inplace=False)

    def forward(self, x471):
        x472=self.layernorm42(x471)
        x473=self.linear40(x472)
        x474=self.gelu19(x473)
        x475=self.dropout38(x474)
        x476=self.linear41(x475)
        x477=self.dropout39(x476)
        return x477

m = M().eval()
x471 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x471)
end = time.time()
print(end-start)
