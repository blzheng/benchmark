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
        self.relu61 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d50 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU(inplace=True)

    def forward(self, x251):
        x252=self.relu61(x251)
        x253=self.conv2d80(x252)
        x254=self.batchnorm2d50(x253)
        x255=self.relu62(x254)
        return x255

m = M().eval()
x251 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
