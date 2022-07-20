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
        self.linear44 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.linear45 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x264):
        x265=self.linear44(x264)
        x266=self.gelu22(x265)
        x267=self.linear45(x266)
        return x267

m = M().eval()
x264 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x264)
end = time.time()
print(end-start)
