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
        self.conv2d15 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d13 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)

    def forward(self, x45):
        x46=self.conv2d15(x45)
        x47=self.batchnorm2d13(x46)
        x48=self.relu9(x47)
        return x48

m = M().eval()
x45 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
