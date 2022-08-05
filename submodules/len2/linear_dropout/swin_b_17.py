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
        self.linear37 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout35 = Dropout(p=0.0, inplace=False)

    def forward(self, x429):
        x430=self.linear37(x429)
        x431=self.dropout35(x430)
        return x431

m = M().eval()
x429 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x429)
end = time.time()
print(end-start)
