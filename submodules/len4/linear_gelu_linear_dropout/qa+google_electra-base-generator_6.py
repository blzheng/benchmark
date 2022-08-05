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
        self.linear41 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear42 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout21 = Dropout(p=0.1, inplace=False)

    def forward(self, x317):
        x318=self.linear41(x317)
        x319=torch._C._nn.gelu(x318)
        x320=self.linear42(x319)
        x321=self.dropout21(x320)
        return x321

m = M().eval()
x317 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x317)
end = time.time()
print(end-start)
