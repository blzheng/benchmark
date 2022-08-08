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
        self.linear45 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout23 = Dropout(p=0.1, inplace=False)
        self.layernorm15 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x350, x353, x322):
        x354=x350.view(x353)
        x355=self.linear45(x354)
        x356=self.dropout23(x355)
        x357=operator.add(x356, x322)
        x358=self.layernorm15(x357)
        return x358

m = M().eval()
x350 = torch.randn(torch.Size([1, 384, 12, 64]))
x353 = (1, 384, 768, )
x322 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x350, x353, x322)
end = time.time()
print(end-start)
