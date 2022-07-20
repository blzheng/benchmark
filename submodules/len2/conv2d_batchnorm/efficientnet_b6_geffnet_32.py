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
        self.conv2d54 = Conv2d(432, 432, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=432, bias=False)
        self.batchnorm2d32 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x162):
        x163=self.conv2d54(x162)
        x164=self.batchnorm2d32(x163)
        return x164

m = M().eval()
x162 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x162)
end = time.time()
print(end-start)