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
        self.linear36 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.dropout34 = Dropout(p=0.0, inplace=False)
        self.linear37 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x426):
        x427=self.linear36(x426)
        x428=self.gelu17(x427)
        x429=self.dropout34(x428)
        x430=self.linear37(x429)
        return x430

m = M().eval()
x426 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x426)
end = time.time()
print(end-start)
