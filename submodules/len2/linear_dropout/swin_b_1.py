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
        self.linear3 = Linear(in_features=512, out_features=128, bias=True)
        self.dropout3 = Dropout(p=0.0, inplace=False)

    def forward(self, x45):
        x46=self.linear3(x45)
        x47=self.dropout3(x46)
        return x47

m = M().eval()
x45 = torch.randn(torch.Size([1, 56, 56, 512]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
