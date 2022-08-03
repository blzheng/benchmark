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
        self.conv2d0 = Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
        self.layernorm0 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm1 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm2 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=512, out_features=128, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)
        self.layernorm3 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm4 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)
        self.linear3 = Linear(in_features=512, out_features=128, bias=True)
        self.dropout3 = Dropout(p=0.0, inplace=False)
        self.layernorm5 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=512, out_features=256, bias=False)
        self.layernorm6 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.layernorm7 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.linear5 = Linear(in_features=256, out_features=1024, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.dropout4 = Dropout(p=0.0, inplace=False)
        self.linear6 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.0, inplace=False)
        self.layernorm8 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.layernorm9 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=256, out_features=1024, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.dropout6 = Dropout(p=0.0, inplace=False)
        self.linear8 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout7 = Dropout(p=0.0, inplace=False)
        self.layernorm10 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear9 = Linear(in_features=1024, out_features=512, bias=False)
        self.layernorm11 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm12 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.dropout8 = Dropout(p=0.0, inplace=False)
        self.linear11 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout9 = Dropout(p=0.0, inplace=False)
        self.layernorm13 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm14 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)
        self.layernorm15 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm16 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear14 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout13 = Dropout(p=0.0, inplace=False)
        self.layernorm17 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm18 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)
        self.layernorm19 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm20 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)
        self.layernorm21 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm22 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)
        self.layernorm23 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm24 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear23 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)
        self.layernorm25 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm26 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear25 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)
        self.layernorm27 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm28 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear26 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout25 = Dropout(p=0.0, inplace=False)
        self.layernorm29 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm30 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.dropout26 = Dropout(p=0.0, inplace=False)
        self.linear29 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout27 = Dropout(p=0.0, inplace=False)
        self.layernorm31 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm32 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.dropout28 = Dropout(p=0.0, inplace=False)
        self.linear31 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout29 = Dropout(p=0.0, inplace=False)
        self.layernorm33 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm34 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear32 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu15 = GELU(approximate='none')
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout31 = Dropout(p=0.0, inplace=False)
        self.layernorm35 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm36 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.dropout32 = Dropout(p=0.0, inplace=False)
        self.linear35 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout33 = Dropout(p=0.0, inplace=False)
        self.layernorm37 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm38 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.dropout34 = Dropout(p=0.0, inplace=False)
        self.linear37 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout35 = Dropout(p=0.0, inplace=False)
        self.layernorm39 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm40 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear38 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu18 = GELU(approximate='none')
        self.dropout36 = Dropout(p=0.0, inplace=False)
        self.linear39 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout37 = Dropout(p=0.0, inplace=False)
        self.layernorm41 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm42 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu19 = GELU(approximate='none')
        self.dropout38 = Dropout(p=0.0, inplace=False)
        self.linear41 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout39 = Dropout(p=0.0, inplace=False)
        self.layernorm43 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm44 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear42 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu20 = GELU(approximate='none')
        self.dropout40 = Dropout(p=0.0, inplace=False)
        self.linear43 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout41 = Dropout(p=0.0, inplace=False)
        self.layernorm45 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm46 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear44 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu21 = GELU(approximate='none')
        self.dropout42 = Dropout(p=0.0, inplace=False)
        self.linear45 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout43 = Dropout(p=0.0, inplace=False)
        self.layernorm47 = LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        self.linear46 = Linear(in_features=2048, out_features=1024, bias=False)
        self.layernorm48 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.layernorm49 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear47 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.dropout44 = Dropout(p=0.0, inplace=False)
        self.linear48 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout45 = Dropout(p=0.0, inplace=False)
        self.layernorm50 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.layernorm51 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear49 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu23 = GELU(approximate='none')
        self.dropout46 = Dropout(p=0.0, inplace=False)
        self.linear50 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout47 = Dropout(p=0.0, inplace=False)
        self.layernorm52 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear51 = Linear(in_features=1024, out_features=1000, bias=True)
        self.relative_position_bias_table0 = torch.rand(torch.Size([169, 4])).to(torch.float32)
        self.relative_position_index0 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight0 = torch.rand(torch.Size([384, 128])).to(torch.float32)
        self.weight1 = torch.rand(torch.Size([128, 128])).to(torch.float32)
        self.bias0 = torch.rand(torch.Size([384])).to(torch.float32)
        self.bias1 = torch.rand(torch.Size([128])).to(torch.float32)
        self.relative_position_bias_table1 = torch.rand(torch.Size([169, 4])).to(torch.float32)
        self.relative_position_index1 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight2 = torch.rand(torch.Size([384, 128])).to(torch.float32)
        self.weight3 = torch.rand(torch.Size([128, 128])).to(torch.float32)
        self.bias2 = torch.rand(torch.Size([384])).to(torch.float32)
        self.bias3 = torch.rand(torch.Size([128])).to(torch.float32)
        self.relative_position_bias_table2 = torch.rand(torch.Size([169, 8])).to(torch.float32)
        self.relative_position_index2 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight4 = torch.rand(torch.Size([768, 256])).to(torch.float32)
        self.weight5 = torch.rand(torch.Size([256, 256])).to(torch.float32)
        self.bias4 = torch.rand(torch.Size([768])).to(torch.float32)
        self.bias5 = torch.rand(torch.Size([256])).to(torch.float32)
        self.relative_position_bias_table3 = torch.rand(torch.Size([169, 8])).to(torch.float32)
        self.relative_position_index3 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight6 = torch.rand(torch.Size([768, 256])).to(torch.float32)
        self.weight7 = torch.rand(torch.Size([256, 256])).to(torch.float32)
        self.bias6 = torch.rand(torch.Size([768])).to(torch.float32)
        self.bias7 = torch.rand(torch.Size([256])).to(torch.float32)
        self.relative_position_bias_table4 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index4 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight8 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight9 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias8 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias9 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table5 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index5 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight10 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight11 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias10 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias11 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table6 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index6 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight12 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight13 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias12 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias13 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table7 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index7 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight14 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight15 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias14 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias15 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table8 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index8 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight16 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight17 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias16 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias17 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table9 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index9 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight18 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight19 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias18 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias19 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table10 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index10 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight20 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight21 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias20 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias21 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table11 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index11 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight22 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight23 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias22 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias23 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table12 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index12 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight24 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight25 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias24 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias25 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table13 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index13 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight26 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight27 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias26 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias27 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table14 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index14 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight28 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight29 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias28 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias29 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table15 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index15 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight30 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight31 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias30 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias31 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table16 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index16 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight32 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight33 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias32 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias33 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table17 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index17 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight34 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight35 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias34 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias35 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table18 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index18 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight36 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight37 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias36 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias37 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table19 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index19 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight38 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight39 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias38 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias39 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table20 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index20 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight40 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight41 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias40 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias41 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table21 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index21 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight42 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight43 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias42 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias43 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table22 = torch.rand(torch.Size([169, 32])).to(torch.float32)
        self.relative_position_index22 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight44 = torch.rand(torch.Size([3072, 1024])).to(torch.float32)
        self.weight45 = torch.rand(torch.Size([1024, 1024])).to(torch.float32)
        self.bias44 = torch.rand(torch.Size([3072])).to(torch.float32)
        self.bias45 = torch.rand(torch.Size([1024])).to(torch.float32)
        self.relative_position_bias_table23 = torch.rand(torch.Size([169, 32])).to(torch.float32)
        self.relative_position_index23 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight46 = torch.rand(torch.Size([3072, 1024])).to(torch.float32)
        self.weight47 = torch.rand(torch.Size([1024, 1024])).to(torch.float32)
        self.bias46 = torch.rand(torch.Size([3072])).to(torch.float32)
        self.bias47 = torch.rand(torch.Size([1024])).to(torch.float32)

    def forward(self, x):
        x0=x
        if x0 is None:
            print('x0: {}'.format(x0))
        elif isinstance(x0, torch.Tensor):
            print('x0: {}'.format(x0.shape))
        elif isinstance(x0, tuple):
            tuple_shapes = '('
            for item in x0:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x0: {}'.format(tuple_shapes))
        else:
            print('x0: {}'.format(x0))
        x1=self.conv2d0(x0)
        if x1 is None:
            print('x1: {}'.format(x1))
        elif isinstance(x1, torch.Tensor):
            print('x1: {}'.format(x1.shape))
        elif isinstance(x1, tuple):
            tuple_shapes = '('
            for item in x1:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x1: {}'.format(tuple_shapes))
        else:
            print('x1: {}'.format(x1))
        x2=torch.permute(x1, [0, 2, 3, 1])
        if x2 is None:
            print('x2: {}'.format(x2))
        elif isinstance(x2, torch.Tensor):
            print('x2: {}'.format(x2.shape))
        elif isinstance(x2, tuple):
            tuple_shapes = '('
            for item in x2:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x2: {}'.format(tuple_shapes))
        else:
            print('x2: {}'.format(x2))
        x3=self.layernorm0(x2)
        if x3 is None:
            print('x3: {}'.format(x3))
        elif isinstance(x3, torch.Tensor):
            print('x3: {}'.format(x3.shape))
        elif isinstance(x3, tuple):
            tuple_shapes = '('
            for item in x3:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x3: {}'.format(tuple_shapes))
        else:
            print('x3: {}'.format(x3))
        x4=self.layernorm1(x3)
        if x4 is None:
            print('x4: {}'.format(x4))
        elif isinstance(x4, torch.Tensor):
            print('x4: {}'.format(x4.shape))
        elif isinstance(x4, tuple):
            tuple_shapes = '('
            for item in x4:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x4: {}'.format(tuple_shapes))
        else:
            print('x4: {}'.format(x4))
        x7=operator.getitem(self.relative_position_bias_table0, self.relative_position_index0)
        if x7 is None:
            print('x7: {}'.format(x7))
        elif isinstance(x7, torch.Tensor):
            print('x7: {}'.format(x7.shape))
        elif isinstance(x7, tuple):
            tuple_shapes = '('
            for item in x7:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x7: {}'.format(tuple_shapes))
        else:
            print('x7: {}'.format(x7))
        x8=x7.view(49, 49, -1)
        if x8 is None:
            print('x8: {}'.format(x8))
        elif isinstance(x8, torch.Tensor):
            print('x8: {}'.format(x8.shape))
        elif isinstance(x8, tuple):
            tuple_shapes = '('
            for item in x8:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x8: {}'.format(tuple_shapes))
        else:
            print('x8: {}'.format(x8))
        x9=x8.permute(2, 0, 1)
        if x9 is None:
            print('x9: {}'.format(x9))
        elif isinstance(x9, torch.Tensor):
            print('x9: {}'.format(x9.shape))
        elif isinstance(x9, tuple):
            tuple_shapes = '('
            for item in x9:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x9: {}'.format(tuple_shapes))
        else:
            print('x9: {}'.format(x9))
        x10=x9.contiguous()
        if x10 is None:
            print('x10: {}'.format(x10))
        elif isinstance(x10, torch.Tensor):
            print('x10: {}'.format(x10.shape))
        elif isinstance(x10, tuple):
            tuple_shapes = '('
            for item in x10:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x10: {}'.format(tuple_shapes))
        else:
            print('x10: {}'.format(x10))
        x11=x10.unsqueeze(0)
        if x11 is None:
            print('x11: {}'.format(x11))
        elif isinstance(x11, torch.Tensor):
            print('x11: {}'.format(x11.shape))
        elif isinstance(x11, tuple):
            tuple_shapes = '('
            for item in x11:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x11: {}'.format(tuple_shapes))
        else:
            print('x11: {}'.format(x11))
        x16=torchvision.models.swin_transformer.shifted_window_attention(x4, self.weight0, self.weight1, x11, [7, 7], 4,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias0, proj_bias=self.bias1)
        if x16 is None:
            print('x16: {}'.format(x16))
        elif isinstance(x16, torch.Tensor):
            print('x16: {}'.format(x16.shape))
        elif isinstance(x16, tuple):
            tuple_shapes = '('
            for item in x16:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x16: {}'.format(tuple_shapes))
        else:
            print('x16: {}'.format(x16))
        x17=stochastic_depth(x16, 0.0, 'row', False)
        if x17 is None:
            print('x17: {}'.format(x17))
        elif isinstance(x17, torch.Tensor):
            print('x17: {}'.format(x17.shape))
        elif isinstance(x17, tuple):
            tuple_shapes = '('
            for item in x17:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x17: {}'.format(tuple_shapes))
        else:
            print('x17: {}'.format(x17))
        x18=operator.add(x3, x17)
        if x18 is None:
            print('x18: {}'.format(x18))
        elif isinstance(x18, torch.Tensor):
            print('x18: {}'.format(x18.shape))
        elif isinstance(x18, tuple):
            tuple_shapes = '('
            for item in x18:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x18: {}'.format(tuple_shapes))
        else:
            print('x18: {}'.format(x18))
        x19=self.layernorm2(x18)
        if x19 is None:
            print('x19: {}'.format(x19))
        elif isinstance(x19, torch.Tensor):
            print('x19: {}'.format(x19.shape))
        elif isinstance(x19, tuple):
            tuple_shapes = '('
            for item in x19:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x19: {}'.format(tuple_shapes))
        else:
            print('x19: {}'.format(x19))
        x20=self.linear0(x19)
        if x20 is None:
            print('x20: {}'.format(x20))
        elif isinstance(x20, torch.Tensor):
            print('x20: {}'.format(x20.shape))
        elif isinstance(x20, tuple):
            tuple_shapes = '('
            for item in x20:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x20: {}'.format(tuple_shapes))
        else:
            print('x20: {}'.format(x20))
        x21=self.gelu0(x20)
        if x21 is None:
            print('x21: {}'.format(x21))
        elif isinstance(x21, torch.Tensor):
            print('x21: {}'.format(x21.shape))
        elif isinstance(x21, tuple):
            tuple_shapes = '('
            for item in x21:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x21: {}'.format(tuple_shapes))
        else:
            print('x21: {}'.format(x21))
        x22=self.dropout0(x21)
        if x22 is None:
            print('x22: {}'.format(x22))
        elif isinstance(x22, torch.Tensor):
            print('x22: {}'.format(x22.shape))
        elif isinstance(x22, tuple):
            tuple_shapes = '('
            for item in x22:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x22: {}'.format(tuple_shapes))
        else:
            print('x22: {}'.format(x22))
        x23=self.linear1(x22)
        if x23 is None:
            print('x23: {}'.format(x23))
        elif isinstance(x23, torch.Tensor):
            print('x23: {}'.format(x23.shape))
        elif isinstance(x23, tuple):
            tuple_shapes = '('
            for item in x23:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x23: {}'.format(tuple_shapes))
        else:
            print('x23: {}'.format(x23))
        x24=self.dropout1(x23)
        if x24 is None:
            print('x24: {}'.format(x24))
        elif isinstance(x24, torch.Tensor):
            print('x24: {}'.format(x24.shape))
        elif isinstance(x24, tuple):
            tuple_shapes = '('
            for item in x24:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x24: {}'.format(tuple_shapes))
        else:
            print('x24: {}'.format(x24))
        x25=stochastic_depth(x24, 0.0, 'row', False)
        if x25 is None:
            print('x25: {}'.format(x25))
        elif isinstance(x25, torch.Tensor):
            print('x25: {}'.format(x25.shape))
        elif isinstance(x25, tuple):
            tuple_shapes = '('
            for item in x25:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x25: {}'.format(tuple_shapes))
        else:
            print('x25: {}'.format(x25))
        x26=operator.add(x18, x25)
        if x26 is None:
            print('x26: {}'.format(x26))
        elif isinstance(x26, torch.Tensor):
            print('x26: {}'.format(x26.shape))
        elif isinstance(x26, tuple):
            tuple_shapes = '('
            for item in x26:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x26: {}'.format(tuple_shapes))
        else:
            print('x26: {}'.format(x26))
        x27=self.layernorm3(x26)
        if x27 is None:
            print('x27: {}'.format(x27))
        elif isinstance(x27, torch.Tensor):
            print('x27: {}'.format(x27.shape))
        elif isinstance(x27, tuple):
            tuple_shapes = '('
            for item in x27:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x27: {}'.format(tuple_shapes))
        else:
            print('x27: {}'.format(x27))
        x30=operator.getitem(self.relative_position_bias_table1, self.relative_position_index1)
        if x30 is None:
            print('x30: {}'.format(x30))
        elif isinstance(x30, torch.Tensor):
            print('x30: {}'.format(x30.shape))
        elif isinstance(x30, tuple):
            tuple_shapes = '('
            for item in x30:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x30: {}'.format(tuple_shapes))
        else:
            print('x30: {}'.format(x30))
        x31=x30.view(49, 49, -1)
        if x31 is None:
            print('x31: {}'.format(x31))
        elif isinstance(x31, torch.Tensor):
            print('x31: {}'.format(x31.shape))
        elif isinstance(x31, tuple):
            tuple_shapes = '('
            for item in x31:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x31: {}'.format(tuple_shapes))
        else:
            print('x31: {}'.format(x31))
        x32=x31.permute(2, 0, 1)
        if x32 is None:
            print('x32: {}'.format(x32))
        elif isinstance(x32, torch.Tensor):
            print('x32: {}'.format(x32.shape))
        elif isinstance(x32, tuple):
            tuple_shapes = '('
            for item in x32:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x32: {}'.format(tuple_shapes))
        else:
            print('x32: {}'.format(x32))
        x33=x32.contiguous()
        if x33 is None:
            print('x33: {}'.format(x33))
        elif isinstance(x33, torch.Tensor):
            print('x33: {}'.format(x33.shape))
        elif isinstance(x33, tuple):
            tuple_shapes = '('
            for item in x33:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x33: {}'.format(tuple_shapes))
        else:
            print('x33: {}'.format(x33))
        x34=x33.unsqueeze(0)
        if x34 is None:
            print('x34: {}'.format(x34))
        elif isinstance(x34, torch.Tensor):
            print('x34: {}'.format(x34.shape))
        elif isinstance(x34, tuple):
            tuple_shapes = '('
            for item in x34:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x34: {}'.format(tuple_shapes))
        else:
            print('x34: {}'.format(x34))
        x39=torchvision.models.swin_transformer.shifted_window_attention(x27, self.weight2, self.weight3, x34, [7, 7], 4,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias2, proj_bias=self.bias3)
        if x39 is None:
            print('x39: {}'.format(x39))
        elif isinstance(x39, torch.Tensor):
            print('x39: {}'.format(x39.shape))
        elif isinstance(x39, tuple):
            tuple_shapes = '('
            for item in x39:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x39: {}'.format(tuple_shapes))
        else:
            print('x39: {}'.format(x39))
        x40=stochastic_depth(x39, 0.021739130434782608, 'row', False)
        if x40 is None:
            print('x40: {}'.format(x40))
        elif isinstance(x40, torch.Tensor):
            print('x40: {}'.format(x40.shape))
        elif isinstance(x40, tuple):
            tuple_shapes = '('
            for item in x40:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x40: {}'.format(tuple_shapes))
        else:
            print('x40: {}'.format(x40))
        x41=operator.add(x26, x40)
        if x41 is None:
            print('x41: {}'.format(x41))
        elif isinstance(x41, torch.Tensor):
            print('x41: {}'.format(x41.shape))
        elif isinstance(x41, tuple):
            tuple_shapes = '('
            for item in x41:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x41: {}'.format(tuple_shapes))
        else:
            print('x41: {}'.format(x41))
        x42=self.layernorm4(x41)
        if x42 is None:
            print('x42: {}'.format(x42))
        elif isinstance(x42, torch.Tensor):
            print('x42: {}'.format(x42.shape))
        elif isinstance(x42, tuple):
            tuple_shapes = '('
            for item in x42:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x42: {}'.format(tuple_shapes))
        else:
            print('x42: {}'.format(x42))
        x43=self.linear2(x42)
        if x43 is None:
            print('x43: {}'.format(x43))
        elif isinstance(x43, torch.Tensor):
            print('x43: {}'.format(x43.shape))
        elif isinstance(x43, tuple):
            tuple_shapes = '('
            for item in x43:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x43: {}'.format(tuple_shapes))
        else:
            print('x43: {}'.format(x43))
        x44=self.gelu1(x43)
        if x44 is None:
            print('x44: {}'.format(x44))
        elif isinstance(x44, torch.Tensor):
            print('x44: {}'.format(x44.shape))
        elif isinstance(x44, tuple):
            tuple_shapes = '('
            for item in x44:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x44: {}'.format(tuple_shapes))
        else:
            print('x44: {}'.format(x44))
        x45=self.dropout2(x44)
        if x45 is None:
            print('x45: {}'.format(x45))
        elif isinstance(x45, torch.Tensor):
            print('x45: {}'.format(x45.shape))
        elif isinstance(x45, tuple):
            tuple_shapes = '('
            for item in x45:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x45: {}'.format(tuple_shapes))
        else:
            print('x45: {}'.format(x45))
        x46=self.linear3(x45)
        if x46 is None:
            print('x46: {}'.format(x46))
        elif isinstance(x46, torch.Tensor):
            print('x46: {}'.format(x46.shape))
        elif isinstance(x46, tuple):
            tuple_shapes = '('
            for item in x46:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x46: {}'.format(tuple_shapes))
        else:
            print('x46: {}'.format(x46))
        x47=self.dropout3(x46)
        if x47 is None:
            print('x47: {}'.format(x47))
        elif isinstance(x47, torch.Tensor):
            print('x47: {}'.format(x47.shape))
        elif isinstance(x47, tuple):
            tuple_shapes = '('
            for item in x47:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x47: {}'.format(tuple_shapes))
        else:
            print('x47: {}'.format(x47))
        x48=stochastic_depth(x47, 0.021739130434782608, 'row', False)
        if x48 is None:
            print('x48: {}'.format(x48))
        elif isinstance(x48, torch.Tensor):
            print('x48: {}'.format(x48.shape))
        elif isinstance(x48, tuple):
            tuple_shapes = '('
            for item in x48:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x48: {}'.format(tuple_shapes))
        else:
            print('x48: {}'.format(x48))
        x49=operator.add(x41, x48)
        if x49 is None:
            print('x49: {}'.format(x49))
        elif isinstance(x49, torch.Tensor):
            print('x49: {}'.format(x49.shape))
        elif isinstance(x49, tuple):
            tuple_shapes = '('
            for item in x49:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x49: {}'.format(tuple_shapes))
        else:
            print('x49: {}'.format(x49))
        x50=torchvision.models.swin_transformer._patch_merging_pad(x49)
        if x50 is None:
            print('x50: {}'.format(x50))
        elif isinstance(x50, torch.Tensor):
            print('x50: {}'.format(x50.shape))
        elif isinstance(x50, tuple):
            tuple_shapes = '('
            for item in x50:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x50: {}'.format(tuple_shapes))
        else:
            print('x50: {}'.format(x50))
        x51=operator.getitem(x50, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x51 is None:
            print('x51: {}'.format(x51))
        elif isinstance(x51, torch.Tensor):
            print('x51: {}'.format(x51.shape))
        elif isinstance(x51, tuple):
            tuple_shapes = '('
            for item in x51:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x51: {}'.format(tuple_shapes))
        else:
            print('x51: {}'.format(x51))
        x52=operator.getitem(x50, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x52 is None:
            print('x52: {}'.format(x52))
        elif isinstance(x52, torch.Tensor):
            print('x52: {}'.format(x52.shape))
        elif isinstance(x52, tuple):
            tuple_shapes = '('
            for item in x52:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x52: {}'.format(tuple_shapes))
        else:
            print('x52: {}'.format(x52))
        x53=operator.getitem(x50, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x53 is None:
            print('x53: {}'.format(x53))
        elif isinstance(x53, torch.Tensor):
            print('x53: {}'.format(x53.shape))
        elif isinstance(x53, tuple):
            tuple_shapes = '('
            for item in x53:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x53: {}'.format(tuple_shapes))
        else:
            print('x53: {}'.format(x53))
        x54=operator.getitem(x50, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x54 is None:
            print('x54: {}'.format(x54))
        elif isinstance(x54, torch.Tensor):
            print('x54: {}'.format(x54.shape))
        elif isinstance(x54, tuple):
            tuple_shapes = '('
            for item in x54:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x54: {}'.format(tuple_shapes))
        else:
            print('x54: {}'.format(x54))
        x55=torch.cat([x51, x52, x53, x54], -1)
        if x55 is None:
            print('x55: {}'.format(x55))
        elif isinstance(x55, torch.Tensor):
            print('x55: {}'.format(x55.shape))
        elif isinstance(x55, tuple):
            tuple_shapes = '('
            for item in x55:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x55: {}'.format(tuple_shapes))
        else:
            print('x55: {}'.format(x55))
        x56=self.layernorm5(x55)
        if x56 is None:
            print('x56: {}'.format(x56))
        elif isinstance(x56, torch.Tensor):
            print('x56: {}'.format(x56.shape))
        elif isinstance(x56, tuple):
            tuple_shapes = '('
            for item in x56:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x56: {}'.format(tuple_shapes))
        else:
            print('x56: {}'.format(x56))
        x57=self.linear4(x56)
        if x57 is None:
            print('x57: {}'.format(x57))
        elif isinstance(x57, torch.Tensor):
            print('x57: {}'.format(x57.shape))
        elif isinstance(x57, tuple):
            tuple_shapes = '('
            for item in x57:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x57: {}'.format(tuple_shapes))
        else:
            print('x57: {}'.format(x57))
        x58=self.layernorm6(x57)
        if x58 is None:
            print('x58: {}'.format(x58))
        elif isinstance(x58, torch.Tensor):
            print('x58: {}'.format(x58.shape))
        elif isinstance(x58, tuple):
            tuple_shapes = '('
            for item in x58:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x58: {}'.format(tuple_shapes))
        else:
            print('x58: {}'.format(x58))
        x61=operator.getitem(self.relative_position_bias_table2, self.relative_position_index2)
        if x61 is None:
            print('x61: {}'.format(x61))
        elif isinstance(x61, torch.Tensor):
            print('x61: {}'.format(x61.shape))
        elif isinstance(x61, tuple):
            tuple_shapes = '('
            for item in x61:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x61: {}'.format(tuple_shapes))
        else:
            print('x61: {}'.format(x61))
        x62=x61.view(49, 49, -1)
        if x62 is None:
            print('x62: {}'.format(x62))
        elif isinstance(x62, torch.Tensor):
            print('x62: {}'.format(x62.shape))
        elif isinstance(x62, tuple):
            tuple_shapes = '('
            for item in x62:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x62: {}'.format(tuple_shapes))
        else:
            print('x62: {}'.format(x62))
        x63=x62.permute(2, 0, 1)
        if x63 is None:
            print('x63: {}'.format(x63))
        elif isinstance(x63, torch.Tensor):
            print('x63: {}'.format(x63.shape))
        elif isinstance(x63, tuple):
            tuple_shapes = '('
            for item in x63:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x63: {}'.format(tuple_shapes))
        else:
            print('x63: {}'.format(x63))
        x64=x63.contiguous()
        if x64 is None:
            print('x64: {}'.format(x64))
        elif isinstance(x64, torch.Tensor):
            print('x64: {}'.format(x64.shape))
        elif isinstance(x64, tuple):
            tuple_shapes = '('
            for item in x64:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x64: {}'.format(tuple_shapes))
        else:
            print('x64: {}'.format(x64))
        x65=x64.unsqueeze(0)
        if x65 is None:
            print('x65: {}'.format(x65))
        elif isinstance(x65, torch.Tensor):
            print('x65: {}'.format(x65.shape))
        elif isinstance(x65, tuple):
            tuple_shapes = '('
            for item in x65:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x65: {}'.format(tuple_shapes))
        else:
            print('x65: {}'.format(x65))
        x70=torchvision.models.swin_transformer.shifted_window_attention(x58, self.weight4, self.weight5, x65, [7, 7], 8,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias4, proj_bias=self.bias5)
        if x70 is None:
            print('x70: {}'.format(x70))
        elif isinstance(x70, torch.Tensor):
            print('x70: {}'.format(x70.shape))
        elif isinstance(x70, tuple):
            tuple_shapes = '('
            for item in x70:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x70: {}'.format(tuple_shapes))
        else:
            print('x70: {}'.format(x70))
        x71=stochastic_depth(x70, 0.043478260869565216, 'row', False)
        if x71 is None:
            print('x71: {}'.format(x71))
        elif isinstance(x71, torch.Tensor):
            print('x71: {}'.format(x71.shape))
        elif isinstance(x71, tuple):
            tuple_shapes = '('
            for item in x71:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x71: {}'.format(tuple_shapes))
        else:
            print('x71: {}'.format(x71))
        x72=operator.add(x57, x71)
        if x72 is None:
            print('x72: {}'.format(x72))
        elif isinstance(x72, torch.Tensor):
            print('x72: {}'.format(x72.shape))
        elif isinstance(x72, tuple):
            tuple_shapes = '('
            for item in x72:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x72: {}'.format(tuple_shapes))
        else:
            print('x72: {}'.format(x72))
        x73=self.layernorm7(x72)
        if x73 is None:
            print('x73: {}'.format(x73))
        elif isinstance(x73, torch.Tensor):
            print('x73: {}'.format(x73.shape))
        elif isinstance(x73, tuple):
            tuple_shapes = '('
            for item in x73:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x73: {}'.format(tuple_shapes))
        else:
            print('x73: {}'.format(x73))
        x74=self.linear5(x73)
        if x74 is None:
            print('x74: {}'.format(x74))
        elif isinstance(x74, torch.Tensor):
            print('x74: {}'.format(x74.shape))
        elif isinstance(x74, tuple):
            tuple_shapes = '('
            for item in x74:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x74: {}'.format(tuple_shapes))
        else:
            print('x74: {}'.format(x74))
        x75=self.gelu2(x74)
        if x75 is None:
            print('x75: {}'.format(x75))
        elif isinstance(x75, torch.Tensor):
            print('x75: {}'.format(x75.shape))
        elif isinstance(x75, tuple):
            tuple_shapes = '('
            for item in x75:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x75: {}'.format(tuple_shapes))
        else:
            print('x75: {}'.format(x75))
        x76=self.dropout4(x75)
        if x76 is None:
            print('x76: {}'.format(x76))
        elif isinstance(x76, torch.Tensor):
            print('x76: {}'.format(x76.shape))
        elif isinstance(x76, tuple):
            tuple_shapes = '('
            for item in x76:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x76: {}'.format(tuple_shapes))
        else:
            print('x76: {}'.format(x76))
        x77=self.linear6(x76)
        if x77 is None:
            print('x77: {}'.format(x77))
        elif isinstance(x77, torch.Tensor):
            print('x77: {}'.format(x77.shape))
        elif isinstance(x77, tuple):
            tuple_shapes = '('
            for item in x77:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x77: {}'.format(tuple_shapes))
        else:
            print('x77: {}'.format(x77))
        x78=self.dropout5(x77)
        if x78 is None:
            print('x78: {}'.format(x78))
        elif isinstance(x78, torch.Tensor):
            print('x78: {}'.format(x78.shape))
        elif isinstance(x78, tuple):
            tuple_shapes = '('
            for item in x78:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x78: {}'.format(tuple_shapes))
        else:
            print('x78: {}'.format(x78))
        x79=stochastic_depth(x78, 0.043478260869565216, 'row', False)
        if x79 is None:
            print('x79: {}'.format(x79))
        elif isinstance(x79, torch.Tensor):
            print('x79: {}'.format(x79.shape))
        elif isinstance(x79, tuple):
            tuple_shapes = '('
            for item in x79:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x79: {}'.format(tuple_shapes))
        else:
            print('x79: {}'.format(x79))
        x80=operator.add(x72, x79)
        if x80 is None:
            print('x80: {}'.format(x80))
        elif isinstance(x80, torch.Tensor):
            print('x80: {}'.format(x80.shape))
        elif isinstance(x80, tuple):
            tuple_shapes = '('
            for item in x80:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x80: {}'.format(tuple_shapes))
        else:
            print('x80: {}'.format(x80))
        x81=self.layernorm8(x80)
        if x81 is None:
            print('x81: {}'.format(x81))
        elif isinstance(x81, torch.Tensor):
            print('x81: {}'.format(x81.shape))
        elif isinstance(x81, tuple):
            tuple_shapes = '('
            for item in x81:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x81: {}'.format(tuple_shapes))
        else:
            print('x81: {}'.format(x81))
        x84=operator.getitem(self.relative_position_bias_table3, self.relative_position_index3)
        if x84 is None:
            print('x84: {}'.format(x84))
        elif isinstance(x84, torch.Tensor):
            print('x84: {}'.format(x84.shape))
        elif isinstance(x84, tuple):
            tuple_shapes = '('
            for item in x84:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x84: {}'.format(tuple_shapes))
        else:
            print('x84: {}'.format(x84))
        x85=x84.view(49, 49, -1)
        if x85 is None:
            print('x85: {}'.format(x85))
        elif isinstance(x85, torch.Tensor):
            print('x85: {}'.format(x85.shape))
        elif isinstance(x85, tuple):
            tuple_shapes = '('
            for item in x85:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x85: {}'.format(tuple_shapes))
        else:
            print('x85: {}'.format(x85))
        x86=x85.permute(2, 0, 1)
        if x86 is None:
            print('x86: {}'.format(x86))
        elif isinstance(x86, torch.Tensor):
            print('x86: {}'.format(x86.shape))
        elif isinstance(x86, tuple):
            tuple_shapes = '('
            for item in x86:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x86: {}'.format(tuple_shapes))
        else:
            print('x86: {}'.format(x86))
        x87=x86.contiguous()
        if x87 is None:
            print('x87: {}'.format(x87))
        elif isinstance(x87, torch.Tensor):
            print('x87: {}'.format(x87.shape))
        elif isinstance(x87, tuple):
            tuple_shapes = '('
            for item in x87:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x87: {}'.format(tuple_shapes))
        else:
            print('x87: {}'.format(x87))
        x88=x87.unsqueeze(0)
        if x88 is None:
            print('x88: {}'.format(x88))
        elif isinstance(x88, torch.Tensor):
            print('x88: {}'.format(x88.shape))
        elif isinstance(x88, tuple):
            tuple_shapes = '('
            for item in x88:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x88: {}'.format(tuple_shapes))
        else:
            print('x88: {}'.format(x88))
        x93=torchvision.models.swin_transformer.shifted_window_attention(x81, self.weight6, self.weight7, x88, [7, 7], 8,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias6, proj_bias=self.bias7)
        if x93 is None:
            print('x93: {}'.format(x93))
        elif isinstance(x93, torch.Tensor):
            print('x93: {}'.format(x93.shape))
        elif isinstance(x93, tuple):
            tuple_shapes = '('
            for item in x93:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x93: {}'.format(tuple_shapes))
        else:
            print('x93: {}'.format(x93))
        x94=stochastic_depth(x93, 0.06521739130434782, 'row', False)
        if x94 is None:
            print('x94: {}'.format(x94))
        elif isinstance(x94, torch.Tensor):
            print('x94: {}'.format(x94.shape))
        elif isinstance(x94, tuple):
            tuple_shapes = '('
            for item in x94:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x94: {}'.format(tuple_shapes))
        else:
            print('x94: {}'.format(x94))
        x95=operator.add(x80, x94)
        if x95 is None:
            print('x95: {}'.format(x95))
        elif isinstance(x95, torch.Tensor):
            print('x95: {}'.format(x95.shape))
        elif isinstance(x95, tuple):
            tuple_shapes = '('
            for item in x95:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x95: {}'.format(tuple_shapes))
        else:
            print('x95: {}'.format(x95))
        x96=self.layernorm9(x95)
        if x96 is None:
            print('x96: {}'.format(x96))
        elif isinstance(x96, torch.Tensor):
            print('x96: {}'.format(x96.shape))
        elif isinstance(x96, tuple):
            tuple_shapes = '('
            for item in x96:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x96: {}'.format(tuple_shapes))
        else:
            print('x96: {}'.format(x96))
        x97=self.linear7(x96)
        if x97 is None:
            print('x97: {}'.format(x97))
        elif isinstance(x97, torch.Tensor):
            print('x97: {}'.format(x97.shape))
        elif isinstance(x97, tuple):
            tuple_shapes = '('
            for item in x97:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x97: {}'.format(tuple_shapes))
        else:
            print('x97: {}'.format(x97))
        x98=self.gelu3(x97)
        if x98 is None:
            print('x98: {}'.format(x98))
        elif isinstance(x98, torch.Tensor):
            print('x98: {}'.format(x98.shape))
        elif isinstance(x98, tuple):
            tuple_shapes = '('
            for item in x98:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x98: {}'.format(tuple_shapes))
        else:
            print('x98: {}'.format(x98))
        x99=self.dropout6(x98)
        if x99 is None:
            print('x99: {}'.format(x99))
        elif isinstance(x99, torch.Tensor):
            print('x99: {}'.format(x99.shape))
        elif isinstance(x99, tuple):
            tuple_shapes = '('
            for item in x99:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x99: {}'.format(tuple_shapes))
        else:
            print('x99: {}'.format(x99))
        x100=self.linear8(x99)
        if x100 is None:
            print('x100: {}'.format(x100))
        elif isinstance(x100, torch.Tensor):
            print('x100: {}'.format(x100.shape))
        elif isinstance(x100, tuple):
            tuple_shapes = '('
            for item in x100:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x100: {}'.format(tuple_shapes))
        else:
            print('x100: {}'.format(x100))
        x101=self.dropout7(x100)
        if x101 is None:
            print('x101: {}'.format(x101))
        elif isinstance(x101, torch.Tensor):
            print('x101: {}'.format(x101.shape))
        elif isinstance(x101, tuple):
            tuple_shapes = '('
            for item in x101:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x101: {}'.format(tuple_shapes))
        else:
            print('x101: {}'.format(x101))
        x102=stochastic_depth(x101, 0.06521739130434782, 'row', False)
        if x102 is None:
            print('x102: {}'.format(x102))
        elif isinstance(x102, torch.Tensor):
            print('x102: {}'.format(x102.shape))
        elif isinstance(x102, tuple):
            tuple_shapes = '('
            for item in x102:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x102: {}'.format(tuple_shapes))
        else:
            print('x102: {}'.format(x102))
        x103=operator.add(x95, x102)
        if x103 is None:
            print('x103: {}'.format(x103))
        elif isinstance(x103, torch.Tensor):
            print('x103: {}'.format(x103.shape))
        elif isinstance(x103, tuple):
            tuple_shapes = '('
            for item in x103:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x103: {}'.format(tuple_shapes))
        else:
            print('x103: {}'.format(x103))
        x104=torchvision.models.swin_transformer._patch_merging_pad(x103)
        if x104 is None:
            print('x104: {}'.format(x104))
        elif isinstance(x104, torch.Tensor):
            print('x104: {}'.format(x104.shape))
        elif isinstance(x104, tuple):
            tuple_shapes = '('
            for item in x104:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x104: {}'.format(tuple_shapes))
        else:
            print('x104: {}'.format(x104))
        x105=operator.getitem(x104, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x105 is None:
            print('x105: {}'.format(x105))
        elif isinstance(x105, torch.Tensor):
            print('x105: {}'.format(x105.shape))
        elif isinstance(x105, tuple):
            tuple_shapes = '('
            for item in x105:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x105: {}'.format(tuple_shapes))
        else:
            print('x105: {}'.format(x105))
        x106=operator.getitem(x104, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x106 is None:
            print('x106: {}'.format(x106))
        elif isinstance(x106, torch.Tensor):
            print('x106: {}'.format(x106.shape))
        elif isinstance(x106, tuple):
            tuple_shapes = '('
            for item in x106:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x106: {}'.format(tuple_shapes))
        else:
            print('x106: {}'.format(x106))
        x107=operator.getitem(x104, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x107 is None:
            print('x107: {}'.format(x107))
        elif isinstance(x107, torch.Tensor):
            print('x107: {}'.format(x107.shape))
        elif isinstance(x107, tuple):
            tuple_shapes = '('
            for item in x107:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x107: {}'.format(tuple_shapes))
        else:
            print('x107: {}'.format(x107))
        x108=operator.getitem(x104, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x108 is None:
            print('x108: {}'.format(x108))
        elif isinstance(x108, torch.Tensor):
            print('x108: {}'.format(x108.shape))
        elif isinstance(x108, tuple):
            tuple_shapes = '('
            for item in x108:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x108: {}'.format(tuple_shapes))
        else:
            print('x108: {}'.format(x108))
        x109=torch.cat([x105, x106, x107, x108], -1)
        if x109 is None:
            print('x109: {}'.format(x109))
        elif isinstance(x109, torch.Tensor):
            print('x109: {}'.format(x109.shape))
        elif isinstance(x109, tuple):
            tuple_shapes = '('
            for item in x109:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x109: {}'.format(tuple_shapes))
        else:
            print('x109: {}'.format(x109))
        x110=self.layernorm10(x109)
        if x110 is None:
            print('x110: {}'.format(x110))
        elif isinstance(x110, torch.Tensor):
            print('x110: {}'.format(x110.shape))
        elif isinstance(x110, tuple):
            tuple_shapes = '('
            for item in x110:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x110: {}'.format(tuple_shapes))
        else:
            print('x110: {}'.format(x110))
        x111=self.linear9(x110)
        if x111 is None:
            print('x111: {}'.format(x111))
        elif isinstance(x111, torch.Tensor):
            print('x111: {}'.format(x111.shape))
        elif isinstance(x111, tuple):
            tuple_shapes = '('
            for item in x111:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x111: {}'.format(tuple_shapes))
        else:
            print('x111: {}'.format(x111))
        x112=self.layernorm11(x111)
        if x112 is None:
            print('x112: {}'.format(x112))
        elif isinstance(x112, torch.Tensor):
            print('x112: {}'.format(x112.shape))
        elif isinstance(x112, tuple):
            tuple_shapes = '('
            for item in x112:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x112: {}'.format(tuple_shapes))
        else:
            print('x112: {}'.format(x112))
        x115=operator.getitem(self.relative_position_bias_table4, self.relative_position_index4)
        if x115 is None:
            print('x115: {}'.format(x115))
        elif isinstance(x115, torch.Tensor):
            print('x115: {}'.format(x115.shape))
        elif isinstance(x115, tuple):
            tuple_shapes = '('
            for item in x115:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x115: {}'.format(tuple_shapes))
        else:
            print('x115: {}'.format(x115))
        x116=x115.view(49, 49, -1)
        if x116 is None:
            print('x116: {}'.format(x116))
        elif isinstance(x116, torch.Tensor):
            print('x116: {}'.format(x116.shape))
        elif isinstance(x116, tuple):
            tuple_shapes = '('
            for item in x116:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x116: {}'.format(tuple_shapes))
        else:
            print('x116: {}'.format(x116))
        x117=x116.permute(2, 0, 1)
        if x117 is None:
            print('x117: {}'.format(x117))
        elif isinstance(x117, torch.Tensor):
            print('x117: {}'.format(x117.shape))
        elif isinstance(x117, tuple):
            tuple_shapes = '('
            for item in x117:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x117: {}'.format(tuple_shapes))
        else:
            print('x117: {}'.format(x117))
        x118=x117.contiguous()
        if x118 is None:
            print('x118: {}'.format(x118))
        elif isinstance(x118, torch.Tensor):
            print('x118: {}'.format(x118.shape))
        elif isinstance(x118, tuple):
            tuple_shapes = '('
            for item in x118:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x118: {}'.format(tuple_shapes))
        else:
            print('x118: {}'.format(x118))
        x119=x118.unsqueeze(0)
        if x119 is None:
            print('x119: {}'.format(x119))
        elif isinstance(x119, torch.Tensor):
            print('x119: {}'.format(x119.shape))
        elif isinstance(x119, tuple):
            tuple_shapes = '('
            for item in x119:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x119: {}'.format(tuple_shapes))
        else:
            print('x119: {}'.format(x119))
        x124=torchvision.models.swin_transformer.shifted_window_attention(x112, self.weight8, self.weight9, x119, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias8, proj_bias=self.bias9)
        if x124 is None:
            print('x124: {}'.format(x124))
        elif isinstance(x124, torch.Tensor):
            print('x124: {}'.format(x124.shape))
        elif isinstance(x124, tuple):
            tuple_shapes = '('
            for item in x124:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x124: {}'.format(tuple_shapes))
        else:
            print('x124: {}'.format(x124))
        x125=stochastic_depth(x124, 0.08695652173913043, 'row', False)
        if x125 is None:
            print('x125: {}'.format(x125))
        elif isinstance(x125, torch.Tensor):
            print('x125: {}'.format(x125.shape))
        elif isinstance(x125, tuple):
            tuple_shapes = '('
            for item in x125:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x125: {}'.format(tuple_shapes))
        else:
            print('x125: {}'.format(x125))
        x126=operator.add(x111, x125)
        if x126 is None:
            print('x126: {}'.format(x126))
        elif isinstance(x126, torch.Tensor):
            print('x126: {}'.format(x126.shape))
        elif isinstance(x126, tuple):
            tuple_shapes = '('
            for item in x126:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x126: {}'.format(tuple_shapes))
        else:
            print('x126: {}'.format(x126))
        x127=self.layernorm12(x126)
        if x127 is None:
            print('x127: {}'.format(x127))
        elif isinstance(x127, torch.Tensor):
            print('x127: {}'.format(x127.shape))
        elif isinstance(x127, tuple):
            tuple_shapes = '('
            for item in x127:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x127: {}'.format(tuple_shapes))
        else:
            print('x127: {}'.format(x127))
        x128=self.linear10(x127)
        if x128 is None:
            print('x128: {}'.format(x128))
        elif isinstance(x128, torch.Tensor):
            print('x128: {}'.format(x128.shape))
        elif isinstance(x128, tuple):
            tuple_shapes = '('
            for item in x128:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x128: {}'.format(tuple_shapes))
        else:
            print('x128: {}'.format(x128))
        x129=self.gelu4(x128)
        if x129 is None:
            print('x129: {}'.format(x129))
        elif isinstance(x129, torch.Tensor):
            print('x129: {}'.format(x129.shape))
        elif isinstance(x129, tuple):
            tuple_shapes = '('
            for item in x129:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x129: {}'.format(tuple_shapes))
        else:
            print('x129: {}'.format(x129))
        x130=self.dropout8(x129)
        if x130 is None:
            print('x130: {}'.format(x130))
        elif isinstance(x130, torch.Tensor):
            print('x130: {}'.format(x130.shape))
        elif isinstance(x130, tuple):
            tuple_shapes = '('
            for item in x130:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x130: {}'.format(tuple_shapes))
        else:
            print('x130: {}'.format(x130))
        x131=self.linear11(x130)
        if x131 is None:
            print('x131: {}'.format(x131))
        elif isinstance(x131, torch.Tensor):
            print('x131: {}'.format(x131.shape))
        elif isinstance(x131, tuple):
            tuple_shapes = '('
            for item in x131:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x131: {}'.format(tuple_shapes))
        else:
            print('x131: {}'.format(x131))
        x132=self.dropout9(x131)
        if x132 is None:
            print('x132: {}'.format(x132))
        elif isinstance(x132, torch.Tensor):
            print('x132: {}'.format(x132.shape))
        elif isinstance(x132, tuple):
            tuple_shapes = '('
            for item in x132:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x132: {}'.format(tuple_shapes))
        else:
            print('x132: {}'.format(x132))
        x133=stochastic_depth(x132, 0.08695652173913043, 'row', False)
        if x133 is None:
            print('x133: {}'.format(x133))
        elif isinstance(x133, torch.Tensor):
            print('x133: {}'.format(x133.shape))
        elif isinstance(x133, tuple):
            tuple_shapes = '('
            for item in x133:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x133: {}'.format(tuple_shapes))
        else:
            print('x133: {}'.format(x133))
        x134=operator.add(x126, x133)
        if x134 is None:
            print('x134: {}'.format(x134))
        elif isinstance(x134, torch.Tensor):
            print('x134: {}'.format(x134.shape))
        elif isinstance(x134, tuple):
            tuple_shapes = '('
            for item in x134:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x134: {}'.format(tuple_shapes))
        else:
            print('x134: {}'.format(x134))
        x135=self.layernorm13(x134)
        if x135 is None:
            print('x135: {}'.format(x135))
        elif isinstance(x135, torch.Tensor):
            print('x135: {}'.format(x135.shape))
        elif isinstance(x135, tuple):
            tuple_shapes = '('
            for item in x135:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x135: {}'.format(tuple_shapes))
        else:
            print('x135: {}'.format(x135))
        x138=operator.getitem(self.relative_position_bias_table5, self.relative_position_index5)
        if x138 is None:
            print('x138: {}'.format(x138))
        elif isinstance(x138, torch.Tensor):
            print('x138: {}'.format(x138.shape))
        elif isinstance(x138, tuple):
            tuple_shapes = '('
            for item in x138:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x138: {}'.format(tuple_shapes))
        else:
            print('x138: {}'.format(x138))
        x139=x138.view(49, 49, -1)
        if x139 is None:
            print('x139: {}'.format(x139))
        elif isinstance(x139, torch.Tensor):
            print('x139: {}'.format(x139.shape))
        elif isinstance(x139, tuple):
            tuple_shapes = '('
            for item in x139:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x139: {}'.format(tuple_shapes))
        else:
            print('x139: {}'.format(x139))
        x140=x139.permute(2, 0, 1)
        if x140 is None:
            print('x140: {}'.format(x140))
        elif isinstance(x140, torch.Tensor):
            print('x140: {}'.format(x140.shape))
        elif isinstance(x140, tuple):
            tuple_shapes = '('
            for item in x140:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x140: {}'.format(tuple_shapes))
        else:
            print('x140: {}'.format(x140))
        x141=x140.contiguous()
        if x141 is None:
            print('x141: {}'.format(x141))
        elif isinstance(x141, torch.Tensor):
            print('x141: {}'.format(x141.shape))
        elif isinstance(x141, tuple):
            tuple_shapes = '('
            for item in x141:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x141: {}'.format(tuple_shapes))
        else:
            print('x141: {}'.format(x141))
        x142=x141.unsqueeze(0)
        if x142 is None:
            print('x142: {}'.format(x142))
        elif isinstance(x142, torch.Tensor):
            print('x142: {}'.format(x142.shape))
        elif isinstance(x142, tuple):
            tuple_shapes = '('
            for item in x142:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x142: {}'.format(tuple_shapes))
        else:
            print('x142: {}'.format(x142))
        x147=torchvision.models.swin_transformer.shifted_window_attention(x135, self.weight10, self.weight11, x142, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias10, proj_bias=self.bias11)
        if x147 is None:
            print('x147: {}'.format(x147))
        elif isinstance(x147, torch.Tensor):
            print('x147: {}'.format(x147.shape))
        elif isinstance(x147, tuple):
            tuple_shapes = '('
            for item in x147:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x147: {}'.format(tuple_shapes))
        else:
            print('x147: {}'.format(x147))
        x148=stochastic_depth(x147, 0.10869565217391304, 'row', False)
        if x148 is None:
            print('x148: {}'.format(x148))
        elif isinstance(x148, torch.Tensor):
            print('x148: {}'.format(x148.shape))
        elif isinstance(x148, tuple):
            tuple_shapes = '('
            for item in x148:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x148: {}'.format(tuple_shapes))
        else:
            print('x148: {}'.format(x148))
        x149=operator.add(x134, x148)
        if x149 is None:
            print('x149: {}'.format(x149))
        elif isinstance(x149, torch.Tensor):
            print('x149: {}'.format(x149.shape))
        elif isinstance(x149, tuple):
            tuple_shapes = '('
            for item in x149:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x149: {}'.format(tuple_shapes))
        else:
            print('x149: {}'.format(x149))
        x150=self.layernorm14(x149)
        if x150 is None:
            print('x150: {}'.format(x150))
        elif isinstance(x150, torch.Tensor):
            print('x150: {}'.format(x150.shape))
        elif isinstance(x150, tuple):
            tuple_shapes = '('
            for item in x150:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x150: {}'.format(tuple_shapes))
        else:
            print('x150: {}'.format(x150))
        x151=self.linear12(x150)
        if x151 is None:
            print('x151: {}'.format(x151))
        elif isinstance(x151, torch.Tensor):
            print('x151: {}'.format(x151.shape))
        elif isinstance(x151, tuple):
            tuple_shapes = '('
            for item in x151:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x151: {}'.format(tuple_shapes))
        else:
            print('x151: {}'.format(x151))
        x152=self.gelu5(x151)
        if x152 is None:
            print('x152: {}'.format(x152))
        elif isinstance(x152, torch.Tensor):
            print('x152: {}'.format(x152.shape))
        elif isinstance(x152, tuple):
            tuple_shapes = '('
            for item in x152:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x152: {}'.format(tuple_shapes))
        else:
            print('x152: {}'.format(x152))
        x153=self.dropout10(x152)
        if x153 is None:
            print('x153: {}'.format(x153))
        elif isinstance(x153, torch.Tensor):
            print('x153: {}'.format(x153.shape))
        elif isinstance(x153, tuple):
            tuple_shapes = '('
            for item in x153:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x153: {}'.format(tuple_shapes))
        else:
            print('x153: {}'.format(x153))
        x154=self.linear13(x153)
        if x154 is None:
            print('x154: {}'.format(x154))
        elif isinstance(x154, torch.Tensor):
            print('x154: {}'.format(x154.shape))
        elif isinstance(x154, tuple):
            tuple_shapes = '('
            for item in x154:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x154: {}'.format(tuple_shapes))
        else:
            print('x154: {}'.format(x154))
        x155=self.dropout11(x154)
        if x155 is None:
            print('x155: {}'.format(x155))
        elif isinstance(x155, torch.Tensor):
            print('x155: {}'.format(x155.shape))
        elif isinstance(x155, tuple):
            tuple_shapes = '('
            for item in x155:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x155: {}'.format(tuple_shapes))
        else:
            print('x155: {}'.format(x155))
        x156=stochastic_depth(x155, 0.10869565217391304, 'row', False)
        if x156 is None:
            print('x156: {}'.format(x156))
        elif isinstance(x156, torch.Tensor):
            print('x156: {}'.format(x156.shape))
        elif isinstance(x156, tuple):
            tuple_shapes = '('
            for item in x156:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x156: {}'.format(tuple_shapes))
        else:
            print('x156: {}'.format(x156))
        x157=operator.add(x149, x156)
        if x157 is None:
            print('x157: {}'.format(x157))
        elif isinstance(x157, torch.Tensor):
            print('x157: {}'.format(x157.shape))
        elif isinstance(x157, tuple):
            tuple_shapes = '('
            for item in x157:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x157: {}'.format(tuple_shapes))
        else:
            print('x157: {}'.format(x157))
        x158=self.layernorm15(x157)
        if x158 is None:
            print('x158: {}'.format(x158))
        elif isinstance(x158, torch.Tensor):
            print('x158: {}'.format(x158.shape))
        elif isinstance(x158, tuple):
            tuple_shapes = '('
            for item in x158:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x158: {}'.format(tuple_shapes))
        else:
            print('x158: {}'.format(x158))
        x161=operator.getitem(self.relative_position_bias_table6, self.relative_position_index6)
        if x161 is None:
            print('x161: {}'.format(x161))
        elif isinstance(x161, torch.Tensor):
            print('x161: {}'.format(x161.shape))
        elif isinstance(x161, tuple):
            tuple_shapes = '('
            for item in x161:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x161: {}'.format(tuple_shapes))
        else:
            print('x161: {}'.format(x161))
        x162=x161.view(49, 49, -1)
        if x162 is None:
            print('x162: {}'.format(x162))
        elif isinstance(x162, torch.Tensor):
            print('x162: {}'.format(x162.shape))
        elif isinstance(x162, tuple):
            tuple_shapes = '('
            for item in x162:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x162: {}'.format(tuple_shapes))
        else:
            print('x162: {}'.format(x162))
        x163=x162.permute(2, 0, 1)
        if x163 is None:
            print('x163: {}'.format(x163))
        elif isinstance(x163, torch.Tensor):
            print('x163: {}'.format(x163.shape))
        elif isinstance(x163, tuple):
            tuple_shapes = '('
            for item in x163:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x163: {}'.format(tuple_shapes))
        else:
            print('x163: {}'.format(x163))
        x164=x163.contiguous()
        if x164 is None:
            print('x164: {}'.format(x164))
        elif isinstance(x164, torch.Tensor):
            print('x164: {}'.format(x164.shape))
        elif isinstance(x164, tuple):
            tuple_shapes = '('
            for item in x164:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x164: {}'.format(tuple_shapes))
        else:
            print('x164: {}'.format(x164))
        x165=x164.unsqueeze(0)
        if x165 is None:
            print('x165: {}'.format(x165))
        elif isinstance(x165, torch.Tensor):
            print('x165: {}'.format(x165.shape))
        elif isinstance(x165, tuple):
            tuple_shapes = '('
            for item in x165:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x165: {}'.format(tuple_shapes))
        else:
            print('x165: {}'.format(x165))
        x170=torchvision.models.swin_transformer.shifted_window_attention(x158, self.weight12, self.weight13, x165, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias12, proj_bias=self.bias13)
        if x170 is None:
            print('x170: {}'.format(x170))
        elif isinstance(x170, torch.Tensor):
            print('x170: {}'.format(x170.shape))
        elif isinstance(x170, tuple):
            tuple_shapes = '('
            for item in x170:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x170: {}'.format(tuple_shapes))
        else:
            print('x170: {}'.format(x170))
        x171=stochastic_depth(x170, 0.13043478260869565, 'row', False)
        if x171 is None:
            print('x171: {}'.format(x171))
        elif isinstance(x171, torch.Tensor):
            print('x171: {}'.format(x171.shape))
        elif isinstance(x171, tuple):
            tuple_shapes = '('
            for item in x171:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x171: {}'.format(tuple_shapes))
        else:
            print('x171: {}'.format(x171))
        x172=operator.add(x157, x171)
        if x172 is None:
            print('x172: {}'.format(x172))
        elif isinstance(x172, torch.Tensor):
            print('x172: {}'.format(x172.shape))
        elif isinstance(x172, tuple):
            tuple_shapes = '('
            for item in x172:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x172: {}'.format(tuple_shapes))
        else:
            print('x172: {}'.format(x172))
        x173=self.layernorm16(x172)
        if x173 is None:
            print('x173: {}'.format(x173))
        elif isinstance(x173, torch.Tensor):
            print('x173: {}'.format(x173.shape))
        elif isinstance(x173, tuple):
            tuple_shapes = '('
            for item in x173:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x173: {}'.format(tuple_shapes))
        else:
            print('x173: {}'.format(x173))
        x174=self.linear14(x173)
        if x174 is None:
            print('x174: {}'.format(x174))
        elif isinstance(x174, torch.Tensor):
            print('x174: {}'.format(x174.shape))
        elif isinstance(x174, tuple):
            tuple_shapes = '('
            for item in x174:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x174: {}'.format(tuple_shapes))
        else:
            print('x174: {}'.format(x174))
        x175=self.gelu6(x174)
        if x175 is None:
            print('x175: {}'.format(x175))
        elif isinstance(x175, torch.Tensor):
            print('x175: {}'.format(x175.shape))
        elif isinstance(x175, tuple):
            tuple_shapes = '('
            for item in x175:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x175: {}'.format(tuple_shapes))
        else:
            print('x175: {}'.format(x175))
        x176=self.dropout12(x175)
        if x176 is None:
            print('x176: {}'.format(x176))
        elif isinstance(x176, torch.Tensor):
            print('x176: {}'.format(x176.shape))
        elif isinstance(x176, tuple):
            tuple_shapes = '('
            for item in x176:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x176: {}'.format(tuple_shapes))
        else:
            print('x176: {}'.format(x176))
        x177=self.linear15(x176)
        if x177 is None:
            print('x177: {}'.format(x177))
        elif isinstance(x177, torch.Tensor):
            print('x177: {}'.format(x177.shape))
        elif isinstance(x177, tuple):
            tuple_shapes = '('
            for item in x177:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x177: {}'.format(tuple_shapes))
        else:
            print('x177: {}'.format(x177))
        x178=self.dropout13(x177)
        if x178 is None:
            print('x178: {}'.format(x178))
        elif isinstance(x178, torch.Tensor):
            print('x178: {}'.format(x178.shape))
        elif isinstance(x178, tuple):
            tuple_shapes = '('
            for item in x178:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x178: {}'.format(tuple_shapes))
        else:
            print('x178: {}'.format(x178))
        x179=stochastic_depth(x178, 0.13043478260869565, 'row', False)
        if x179 is None:
            print('x179: {}'.format(x179))
        elif isinstance(x179, torch.Tensor):
            print('x179: {}'.format(x179.shape))
        elif isinstance(x179, tuple):
            tuple_shapes = '('
            for item in x179:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x179: {}'.format(tuple_shapes))
        else:
            print('x179: {}'.format(x179))
        x180=operator.add(x172, x179)
        if x180 is None:
            print('x180: {}'.format(x180))
        elif isinstance(x180, torch.Tensor):
            print('x180: {}'.format(x180.shape))
        elif isinstance(x180, tuple):
            tuple_shapes = '('
            for item in x180:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x180: {}'.format(tuple_shapes))
        else:
            print('x180: {}'.format(x180))
        x181=self.layernorm17(x180)
        if x181 is None:
            print('x181: {}'.format(x181))
        elif isinstance(x181, torch.Tensor):
            print('x181: {}'.format(x181.shape))
        elif isinstance(x181, tuple):
            tuple_shapes = '('
            for item in x181:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x181: {}'.format(tuple_shapes))
        else:
            print('x181: {}'.format(x181))
        x184=operator.getitem(self.relative_position_bias_table7, self.relative_position_index7)
        if x184 is None:
            print('x184: {}'.format(x184))
        elif isinstance(x184, torch.Tensor):
            print('x184: {}'.format(x184.shape))
        elif isinstance(x184, tuple):
            tuple_shapes = '('
            for item in x184:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x184: {}'.format(tuple_shapes))
        else:
            print('x184: {}'.format(x184))
        x185=x184.view(49, 49, -1)
        if x185 is None:
            print('x185: {}'.format(x185))
        elif isinstance(x185, torch.Tensor):
            print('x185: {}'.format(x185.shape))
        elif isinstance(x185, tuple):
            tuple_shapes = '('
            for item in x185:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x185: {}'.format(tuple_shapes))
        else:
            print('x185: {}'.format(x185))
        x186=x185.permute(2, 0, 1)
        if x186 is None:
            print('x186: {}'.format(x186))
        elif isinstance(x186, torch.Tensor):
            print('x186: {}'.format(x186.shape))
        elif isinstance(x186, tuple):
            tuple_shapes = '('
            for item in x186:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x186: {}'.format(tuple_shapes))
        else:
            print('x186: {}'.format(x186))
        x187=x186.contiguous()
        if x187 is None:
            print('x187: {}'.format(x187))
        elif isinstance(x187, torch.Tensor):
            print('x187: {}'.format(x187.shape))
        elif isinstance(x187, tuple):
            tuple_shapes = '('
            for item in x187:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x187: {}'.format(tuple_shapes))
        else:
            print('x187: {}'.format(x187))
        x188=x187.unsqueeze(0)
        if x188 is None:
            print('x188: {}'.format(x188))
        elif isinstance(x188, torch.Tensor):
            print('x188: {}'.format(x188.shape))
        elif isinstance(x188, tuple):
            tuple_shapes = '('
            for item in x188:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x188: {}'.format(tuple_shapes))
        else:
            print('x188: {}'.format(x188))
        x193=torchvision.models.swin_transformer.shifted_window_attention(x181, self.weight14, self.weight15, x188, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias14, proj_bias=self.bias15)
        if x193 is None:
            print('x193: {}'.format(x193))
        elif isinstance(x193, torch.Tensor):
            print('x193: {}'.format(x193.shape))
        elif isinstance(x193, tuple):
            tuple_shapes = '('
            for item in x193:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x193: {}'.format(tuple_shapes))
        else:
            print('x193: {}'.format(x193))
        x194=stochastic_depth(x193, 0.15217391304347827, 'row', False)
        if x194 is None:
            print('x194: {}'.format(x194))
        elif isinstance(x194, torch.Tensor):
            print('x194: {}'.format(x194.shape))
        elif isinstance(x194, tuple):
            tuple_shapes = '('
            for item in x194:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x194: {}'.format(tuple_shapes))
        else:
            print('x194: {}'.format(x194))
        x195=operator.add(x180, x194)
        if x195 is None:
            print('x195: {}'.format(x195))
        elif isinstance(x195, torch.Tensor):
            print('x195: {}'.format(x195.shape))
        elif isinstance(x195, tuple):
            tuple_shapes = '('
            for item in x195:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x195: {}'.format(tuple_shapes))
        else:
            print('x195: {}'.format(x195))
        x196=self.layernorm18(x195)
        if x196 is None:
            print('x196: {}'.format(x196))
        elif isinstance(x196, torch.Tensor):
            print('x196: {}'.format(x196.shape))
        elif isinstance(x196, tuple):
            tuple_shapes = '('
            for item in x196:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x196: {}'.format(tuple_shapes))
        else:
            print('x196: {}'.format(x196))
        x197=self.linear16(x196)
        if x197 is None:
            print('x197: {}'.format(x197))
        elif isinstance(x197, torch.Tensor):
            print('x197: {}'.format(x197.shape))
        elif isinstance(x197, tuple):
            tuple_shapes = '('
            for item in x197:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x197: {}'.format(tuple_shapes))
        else:
            print('x197: {}'.format(x197))
        x198=self.gelu7(x197)
        if x198 is None:
            print('x198: {}'.format(x198))
        elif isinstance(x198, torch.Tensor):
            print('x198: {}'.format(x198.shape))
        elif isinstance(x198, tuple):
            tuple_shapes = '('
            for item in x198:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x198: {}'.format(tuple_shapes))
        else:
            print('x198: {}'.format(x198))
        x199=self.dropout14(x198)
        if x199 is None:
            print('x199: {}'.format(x199))
        elif isinstance(x199, torch.Tensor):
            print('x199: {}'.format(x199.shape))
        elif isinstance(x199, tuple):
            tuple_shapes = '('
            for item in x199:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x199: {}'.format(tuple_shapes))
        else:
            print('x199: {}'.format(x199))
        x200=self.linear17(x199)
        if x200 is None:
            print('x200: {}'.format(x200))
        elif isinstance(x200, torch.Tensor):
            print('x200: {}'.format(x200.shape))
        elif isinstance(x200, tuple):
            tuple_shapes = '('
            for item in x200:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x200: {}'.format(tuple_shapes))
        else:
            print('x200: {}'.format(x200))
        x201=self.dropout15(x200)
        if x201 is None:
            print('x201: {}'.format(x201))
        elif isinstance(x201, torch.Tensor):
            print('x201: {}'.format(x201.shape))
        elif isinstance(x201, tuple):
            tuple_shapes = '('
            for item in x201:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x201: {}'.format(tuple_shapes))
        else:
            print('x201: {}'.format(x201))
        x202=stochastic_depth(x201, 0.15217391304347827, 'row', False)
        if x202 is None:
            print('x202: {}'.format(x202))
        elif isinstance(x202, torch.Tensor):
            print('x202: {}'.format(x202.shape))
        elif isinstance(x202, tuple):
            tuple_shapes = '('
            for item in x202:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x202: {}'.format(tuple_shapes))
        else:
            print('x202: {}'.format(x202))
        x203=operator.add(x195, x202)
        if x203 is None:
            print('x203: {}'.format(x203))
        elif isinstance(x203, torch.Tensor):
            print('x203: {}'.format(x203.shape))
        elif isinstance(x203, tuple):
            tuple_shapes = '('
            for item in x203:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x203: {}'.format(tuple_shapes))
        else:
            print('x203: {}'.format(x203))
        x204=self.layernorm19(x203)
        if x204 is None:
            print('x204: {}'.format(x204))
        elif isinstance(x204, torch.Tensor):
            print('x204: {}'.format(x204.shape))
        elif isinstance(x204, tuple):
            tuple_shapes = '('
            for item in x204:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x204: {}'.format(tuple_shapes))
        else:
            print('x204: {}'.format(x204))
        x207=operator.getitem(self.relative_position_bias_table8, self.relative_position_index8)
        if x207 is None:
            print('x207: {}'.format(x207))
        elif isinstance(x207, torch.Tensor):
            print('x207: {}'.format(x207.shape))
        elif isinstance(x207, tuple):
            tuple_shapes = '('
            for item in x207:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x207: {}'.format(tuple_shapes))
        else:
            print('x207: {}'.format(x207))
        x208=x207.view(49, 49, -1)
        if x208 is None:
            print('x208: {}'.format(x208))
        elif isinstance(x208, torch.Tensor):
            print('x208: {}'.format(x208.shape))
        elif isinstance(x208, tuple):
            tuple_shapes = '('
            for item in x208:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x208: {}'.format(tuple_shapes))
        else:
            print('x208: {}'.format(x208))
        x209=x208.permute(2, 0, 1)
        if x209 is None:
            print('x209: {}'.format(x209))
        elif isinstance(x209, torch.Tensor):
            print('x209: {}'.format(x209.shape))
        elif isinstance(x209, tuple):
            tuple_shapes = '('
            for item in x209:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x209: {}'.format(tuple_shapes))
        else:
            print('x209: {}'.format(x209))
        x210=x209.contiguous()
        if x210 is None:
            print('x210: {}'.format(x210))
        elif isinstance(x210, torch.Tensor):
            print('x210: {}'.format(x210.shape))
        elif isinstance(x210, tuple):
            tuple_shapes = '('
            for item in x210:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x210: {}'.format(tuple_shapes))
        else:
            print('x210: {}'.format(x210))
        x211=x210.unsqueeze(0)
        if x211 is None:
            print('x211: {}'.format(x211))
        elif isinstance(x211, torch.Tensor):
            print('x211: {}'.format(x211.shape))
        elif isinstance(x211, tuple):
            tuple_shapes = '('
            for item in x211:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x211: {}'.format(tuple_shapes))
        else:
            print('x211: {}'.format(x211))
        x216=torchvision.models.swin_transformer.shifted_window_attention(x204, self.weight16, self.weight17, x211, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias16, proj_bias=self.bias17)
        if x216 is None:
            print('x216: {}'.format(x216))
        elif isinstance(x216, torch.Tensor):
            print('x216: {}'.format(x216.shape))
        elif isinstance(x216, tuple):
            tuple_shapes = '('
            for item in x216:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x216: {}'.format(tuple_shapes))
        else:
            print('x216: {}'.format(x216))
        x217=stochastic_depth(x216, 0.17391304347826086, 'row', False)
        if x217 is None:
            print('x217: {}'.format(x217))
        elif isinstance(x217, torch.Tensor):
            print('x217: {}'.format(x217.shape))
        elif isinstance(x217, tuple):
            tuple_shapes = '('
            for item in x217:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x217: {}'.format(tuple_shapes))
        else:
            print('x217: {}'.format(x217))
        x218=operator.add(x203, x217)
        if x218 is None:
            print('x218: {}'.format(x218))
        elif isinstance(x218, torch.Tensor):
            print('x218: {}'.format(x218.shape))
        elif isinstance(x218, tuple):
            tuple_shapes = '('
            for item in x218:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x218: {}'.format(tuple_shapes))
        else:
            print('x218: {}'.format(x218))
        x219=self.layernorm20(x218)
        if x219 is None:
            print('x219: {}'.format(x219))
        elif isinstance(x219, torch.Tensor):
            print('x219: {}'.format(x219.shape))
        elif isinstance(x219, tuple):
            tuple_shapes = '('
            for item in x219:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x219: {}'.format(tuple_shapes))
        else:
            print('x219: {}'.format(x219))
        x220=self.linear18(x219)
        if x220 is None:
            print('x220: {}'.format(x220))
        elif isinstance(x220, torch.Tensor):
            print('x220: {}'.format(x220.shape))
        elif isinstance(x220, tuple):
            tuple_shapes = '('
            for item in x220:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x220: {}'.format(tuple_shapes))
        else:
            print('x220: {}'.format(x220))
        x221=self.gelu8(x220)
        if x221 is None:
            print('x221: {}'.format(x221))
        elif isinstance(x221, torch.Tensor):
            print('x221: {}'.format(x221.shape))
        elif isinstance(x221, tuple):
            tuple_shapes = '('
            for item in x221:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x221: {}'.format(tuple_shapes))
        else:
            print('x221: {}'.format(x221))
        x222=self.dropout16(x221)
        if x222 is None:
            print('x222: {}'.format(x222))
        elif isinstance(x222, torch.Tensor):
            print('x222: {}'.format(x222.shape))
        elif isinstance(x222, tuple):
            tuple_shapes = '('
            for item in x222:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x222: {}'.format(tuple_shapes))
        else:
            print('x222: {}'.format(x222))
        x223=self.linear19(x222)
        if x223 is None:
            print('x223: {}'.format(x223))
        elif isinstance(x223, torch.Tensor):
            print('x223: {}'.format(x223.shape))
        elif isinstance(x223, tuple):
            tuple_shapes = '('
            for item in x223:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x223: {}'.format(tuple_shapes))
        else:
            print('x223: {}'.format(x223))
        x224=self.dropout17(x223)
        if x224 is None:
            print('x224: {}'.format(x224))
        elif isinstance(x224, torch.Tensor):
            print('x224: {}'.format(x224.shape))
        elif isinstance(x224, tuple):
            tuple_shapes = '('
            for item in x224:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x224: {}'.format(tuple_shapes))
        else:
            print('x224: {}'.format(x224))
        x225=stochastic_depth(x224, 0.17391304347826086, 'row', False)
        if x225 is None:
            print('x225: {}'.format(x225))
        elif isinstance(x225, torch.Tensor):
            print('x225: {}'.format(x225.shape))
        elif isinstance(x225, tuple):
            tuple_shapes = '('
            for item in x225:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x225: {}'.format(tuple_shapes))
        else:
            print('x225: {}'.format(x225))
        x226=operator.add(x218, x225)
        if x226 is None:
            print('x226: {}'.format(x226))
        elif isinstance(x226, torch.Tensor):
            print('x226: {}'.format(x226.shape))
        elif isinstance(x226, tuple):
            tuple_shapes = '('
            for item in x226:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x226: {}'.format(tuple_shapes))
        else:
            print('x226: {}'.format(x226))
        x227=self.layernorm21(x226)
        if x227 is None:
            print('x227: {}'.format(x227))
        elif isinstance(x227, torch.Tensor):
            print('x227: {}'.format(x227.shape))
        elif isinstance(x227, tuple):
            tuple_shapes = '('
            for item in x227:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x227: {}'.format(tuple_shapes))
        else:
            print('x227: {}'.format(x227))
        x230=operator.getitem(self.relative_position_bias_table9, self.relative_position_index9)
        if x230 is None:
            print('x230: {}'.format(x230))
        elif isinstance(x230, torch.Tensor):
            print('x230: {}'.format(x230.shape))
        elif isinstance(x230, tuple):
            tuple_shapes = '('
            for item in x230:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x230: {}'.format(tuple_shapes))
        else:
            print('x230: {}'.format(x230))
        x231=x230.view(49, 49, -1)
        if x231 is None:
            print('x231: {}'.format(x231))
        elif isinstance(x231, torch.Tensor):
            print('x231: {}'.format(x231.shape))
        elif isinstance(x231, tuple):
            tuple_shapes = '('
            for item in x231:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x231: {}'.format(tuple_shapes))
        else:
            print('x231: {}'.format(x231))
        x232=x231.permute(2, 0, 1)
        if x232 is None:
            print('x232: {}'.format(x232))
        elif isinstance(x232, torch.Tensor):
            print('x232: {}'.format(x232.shape))
        elif isinstance(x232, tuple):
            tuple_shapes = '('
            for item in x232:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x232: {}'.format(tuple_shapes))
        else:
            print('x232: {}'.format(x232))
        x233=x232.contiguous()
        if x233 is None:
            print('x233: {}'.format(x233))
        elif isinstance(x233, torch.Tensor):
            print('x233: {}'.format(x233.shape))
        elif isinstance(x233, tuple):
            tuple_shapes = '('
            for item in x233:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x233: {}'.format(tuple_shapes))
        else:
            print('x233: {}'.format(x233))
        x234=x233.unsqueeze(0)
        if x234 is None:
            print('x234: {}'.format(x234))
        elif isinstance(x234, torch.Tensor):
            print('x234: {}'.format(x234.shape))
        elif isinstance(x234, tuple):
            tuple_shapes = '('
            for item in x234:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x234: {}'.format(tuple_shapes))
        else:
            print('x234: {}'.format(x234))
        x239=torchvision.models.swin_transformer.shifted_window_attention(x227, self.weight18, self.weight19, x234, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias18, proj_bias=self.bias19)
        if x239 is None:
            print('x239: {}'.format(x239))
        elif isinstance(x239, torch.Tensor):
            print('x239: {}'.format(x239.shape))
        elif isinstance(x239, tuple):
            tuple_shapes = '('
            for item in x239:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x239: {}'.format(tuple_shapes))
        else:
            print('x239: {}'.format(x239))
        x240=stochastic_depth(x239, 0.1956521739130435, 'row', False)
        if x240 is None:
            print('x240: {}'.format(x240))
        elif isinstance(x240, torch.Tensor):
            print('x240: {}'.format(x240.shape))
        elif isinstance(x240, tuple):
            tuple_shapes = '('
            for item in x240:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x240: {}'.format(tuple_shapes))
        else:
            print('x240: {}'.format(x240))
        x241=operator.add(x226, x240)
        if x241 is None:
            print('x241: {}'.format(x241))
        elif isinstance(x241, torch.Tensor):
            print('x241: {}'.format(x241.shape))
        elif isinstance(x241, tuple):
            tuple_shapes = '('
            for item in x241:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x241: {}'.format(tuple_shapes))
        else:
            print('x241: {}'.format(x241))
        x242=self.layernorm22(x241)
        if x242 is None:
            print('x242: {}'.format(x242))
        elif isinstance(x242, torch.Tensor):
            print('x242: {}'.format(x242.shape))
        elif isinstance(x242, tuple):
            tuple_shapes = '('
            for item in x242:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x242: {}'.format(tuple_shapes))
        else:
            print('x242: {}'.format(x242))
        x243=self.linear20(x242)
        if x243 is None:
            print('x243: {}'.format(x243))
        elif isinstance(x243, torch.Tensor):
            print('x243: {}'.format(x243.shape))
        elif isinstance(x243, tuple):
            tuple_shapes = '('
            for item in x243:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x243: {}'.format(tuple_shapes))
        else:
            print('x243: {}'.format(x243))
        x244=self.gelu9(x243)
        if x244 is None:
            print('x244: {}'.format(x244))
        elif isinstance(x244, torch.Tensor):
            print('x244: {}'.format(x244.shape))
        elif isinstance(x244, tuple):
            tuple_shapes = '('
            for item in x244:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x244: {}'.format(tuple_shapes))
        else:
            print('x244: {}'.format(x244))
        x245=self.dropout18(x244)
        if x245 is None:
            print('x245: {}'.format(x245))
        elif isinstance(x245, torch.Tensor):
            print('x245: {}'.format(x245.shape))
        elif isinstance(x245, tuple):
            tuple_shapes = '('
            for item in x245:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x245: {}'.format(tuple_shapes))
        else:
            print('x245: {}'.format(x245))
        x246=self.linear21(x245)
        if x246 is None:
            print('x246: {}'.format(x246))
        elif isinstance(x246, torch.Tensor):
            print('x246: {}'.format(x246.shape))
        elif isinstance(x246, tuple):
            tuple_shapes = '('
            for item in x246:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x246: {}'.format(tuple_shapes))
        else:
            print('x246: {}'.format(x246))
        x247=self.dropout19(x246)
        if x247 is None:
            print('x247: {}'.format(x247))
        elif isinstance(x247, torch.Tensor):
            print('x247: {}'.format(x247.shape))
        elif isinstance(x247, tuple):
            tuple_shapes = '('
            for item in x247:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x247: {}'.format(tuple_shapes))
        else:
            print('x247: {}'.format(x247))
        x248=stochastic_depth(x247, 0.1956521739130435, 'row', False)
        if x248 is None:
            print('x248: {}'.format(x248))
        elif isinstance(x248, torch.Tensor):
            print('x248: {}'.format(x248.shape))
        elif isinstance(x248, tuple):
            tuple_shapes = '('
            for item in x248:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x248: {}'.format(tuple_shapes))
        else:
            print('x248: {}'.format(x248))
        x249=operator.add(x241, x248)
        if x249 is None:
            print('x249: {}'.format(x249))
        elif isinstance(x249, torch.Tensor):
            print('x249: {}'.format(x249.shape))
        elif isinstance(x249, tuple):
            tuple_shapes = '('
            for item in x249:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x249: {}'.format(tuple_shapes))
        else:
            print('x249: {}'.format(x249))
        x250=self.layernorm23(x249)
        if x250 is None:
            print('x250: {}'.format(x250))
        elif isinstance(x250, torch.Tensor):
            print('x250: {}'.format(x250.shape))
        elif isinstance(x250, tuple):
            tuple_shapes = '('
            for item in x250:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x250: {}'.format(tuple_shapes))
        else:
            print('x250: {}'.format(x250))
        x253=operator.getitem(self.relative_position_bias_table10, self.relative_position_index10)
        if x253 is None:
            print('x253: {}'.format(x253))
        elif isinstance(x253, torch.Tensor):
            print('x253: {}'.format(x253.shape))
        elif isinstance(x253, tuple):
            tuple_shapes = '('
            for item in x253:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x253: {}'.format(tuple_shapes))
        else:
            print('x253: {}'.format(x253))
        x254=x253.view(49, 49, -1)
        if x254 is None:
            print('x254: {}'.format(x254))
        elif isinstance(x254, torch.Tensor):
            print('x254: {}'.format(x254.shape))
        elif isinstance(x254, tuple):
            tuple_shapes = '('
            for item in x254:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x254: {}'.format(tuple_shapes))
        else:
            print('x254: {}'.format(x254))
        x255=x254.permute(2, 0, 1)
        if x255 is None:
            print('x255: {}'.format(x255))
        elif isinstance(x255, torch.Tensor):
            print('x255: {}'.format(x255.shape))
        elif isinstance(x255, tuple):
            tuple_shapes = '('
            for item in x255:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x255: {}'.format(tuple_shapes))
        else:
            print('x255: {}'.format(x255))
        x256=x255.contiguous()
        if x256 is None:
            print('x256: {}'.format(x256))
        elif isinstance(x256, torch.Tensor):
            print('x256: {}'.format(x256.shape))
        elif isinstance(x256, tuple):
            tuple_shapes = '('
            for item in x256:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x256: {}'.format(tuple_shapes))
        else:
            print('x256: {}'.format(x256))
        x257=x256.unsqueeze(0)
        if x257 is None:
            print('x257: {}'.format(x257))
        elif isinstance(x257, torch.Tensor):
            print('x257: {}'.format(x257.shape))
        elif isinstance(x257, tuple):
            tuple_shapes = '('
            for item in x257:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x257: {}'.format(tuple_shapes))
        else:
            print('x257: {}'.format(x257))
        x262=torchvision.models.swin_transformer.shifted_window_attention(x250, self.weight20, self.weight21, x257, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias20, proj_bias=self.bias21)
        if x262 is None:
            print('x262: {}'.format(x262))
        elif isinstance(x262, torch.Tensor):
            print('x262: {}'.format(x262.shape))
        elif isinstance(x262, tuple):
            tuple_shapes = '('
            for item in x262:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x262: {}'.format(tuple_shapes))
        else:
            print('x262: {}'.format(x262))
        x263=stochastic_depth(x262, 0.21739130434782608, 'row', False)
        if x263 is None:
            print('x263: {}'.format(x263))
        elif isinstance(x263, torch.Tensor):
            print('x263: {}'.format(x263.shape))
        elif isinstance(x263, tuple):
            tuple_shapes = '('
            for item in x263:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x263: {}'.format(tuple_shapes))
        else:
            print('x263: {}'.format(x263))
        x264=operator.add(x249, x263)
        if x264 is None:
            print('x264: {}'.format(x264))
        elif isinstance(x264, torch.Tensor):
            print('x264: {}'.format(x264.shape))
        elif isinstance(x264, tuple):
            tuple_shapes = '('
            for item in x264:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x264: {}'.format(tuple_shapes))
        else:
            print('x264: {}'.format(x264))
        x265=self.layernorm24(x264)
        if x265 is None:
            print('x265: {}'.format(x265))
        elif isinstance(x265, torch.Tensor):
            print('x265: {}'.format(x265.shape))
        elif isinstance(x265, tuple):
            tuple_shapes = '('
            for item in x265:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x265: {}'.format(tuple_shapes))
        else:
            print('x265: {}'.format(x265))
        x266=self.linear22(x265)
        if x266 is None:
            print('x266: {}'.format(x266))
        elif isinstance(x266, torch.Tensor):
            print('x266: {}'.format(x266.shape))
        elif isinstance(x266, tuple):
            tuple_shapes = '('
            for item in x266:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x266: {}'.format(tuple_shapes))
        else:
            print('x266: {}'.format(x266))
        x267=self.gelu10(x266)
        if x267 is None:
            print('x267: {}'.format(x267))
        elif isinstance(x267, torch.Tensor):
            print('x267: {}'.format(x267.shape))
        elif isinstance(x267, tuple):
            tuple_shapes = '('
            for item in x267:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x267: {}'.format(tuple_shapes))
        else:
            print('x267: {}'.format(x267))
        x268=self.dropout20(x267)
        if x268 is None:
            print('x268: {}'.format(x268))
        elif isinstance(x268, torch.Tensor):
            print('x268: {}'.format(x268.shape))
        elif isinstance(x268, tuple):
            tuple_shapes = '('
            for item in x268:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x268: {}'.format(tuple_shapes))
        else:
            print('x268: {}'.format(x268))
        x269=self.linear23(x268)
        if x269 is None:
            print('x269: {}'.format(x269))
        elif isinstance(x269, torch.Tensor):
            print('x269: {}'.format(x269.shape))
        elif isinstance(x269, tuple):
            tuple_shapes = '('
            for item in x269:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x269: {}'.format(tuple_shapes))
        else:
            print('x269: {}'.format(x269))
        x270=self.dropout21(x269)
        if x270 is None:
            print('x270: {}'.format(x270))
        elif isinstance(x270, torch.Tensor):
            print('x270: {}'.format(x270.shape))
        elif isinstance(x270, tuple):
            tuple_shapes = '('
            for item in x270:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x270: {}'.format(tuple_shapes))
        else:
            print('x270: {}'.format(x270))
        x271=stochastic_depth(x270, 0.21739130434782608, 'row', False)
        if x271 is None:
            print('x271: {}'.format(x271))
        elif isinstance(x271, torch.Tensor):
            print('x271: {}'.format(x271.shape))
        elif isinstance(x271, tuple):
            tuple_shapes = '('
            for item in x271:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x271: {}'.format(tuple_shapes))
        else:
            print('x271: {}'.format(x271))
        x272=operator.add(x264, x271)
        if x272 is None:
            print('x272: {}'.format(x272))
        elif isinstance(x272, torch.Tensor):
            print('x272: {}'.format(x272.shape))
        elif isinstance(x272, tuple):
            tuple_shapes = '('
            for item in x272:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x272: {}'.format(tuple_shapes))
        else:
            print('x272: {}'.format(x272))
        x273=self.layernorm25(x272)
        if x273 is None:
            print('x273: {}'.format(x273))
        elif isinstance(x273, torch.Tensor):
            print('x273: {}'.format(x273.shape))
        elif isinstance(x273, tuple):
            tuple_shapes = '('
            for item in x273:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x273: {}'.format(tuple_shapes))
        else:
            print('x273: {}'.format(x273))
        x276=operator.getitem(self.relative_position_bias_table11, self.relative_position_index11)
        if x276 is None:
            print('x276: {}'.format(x276))
        elif isinstance(x276, torch.Tensor):
            print('x276: {}'.format(x276.shape))
        elif isinstance(x276, tuple):
            tuple_shapes = '('
            for item in x276:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x276: {}'.format(tuple_shapes))
        else:
            print('x276: {}'.format(x276))
        x277=x276.view(49, 49, -1)
        if x277 is None:
            print('x277: {}'.format(x277))
        elif isinstance(x277, torch.Tensor):
            print('x277: {}'.format(x277.shape))
        elif isinstance(x277, tuple):
            tuple_shapes = '('
            for item in x277:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x277: {}'.format(tuple_shapes))
        else:
            print('x277: {}'.format(x277))
        x278=x277.permute(2, 0, 1)
        if x278 is None:
            print('x278: {}'.format(x278))
        elif isinstance(x278, torch.Tensor):
            print('x278: {}'.format(x278.shape))
        elif isinstance(x278, tuple):
            tuple_shapes = '('
            for item in x278:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x278: {}'.format(tuple_shapes))
        else:
            print('x278: {}'.format(x278))
        x279=x278.contiguous()
        if x279 is None:
            print('x279: {}'.format(x279))
        elif isinstance(x279, torch.Tensor):
            print('x279: {}'.format(x279.shape))
        elif isinstance(x279, tuple):
            tuple_shapes = '('
            for item in x279:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x279: {}'.format(tuple_shapes))
        else:
            print('x279: {}'.format(x279))
        x280=x279.unsqueeze(0)
        if x280 is None:
            print('x280: {}'.format(x280))
        elif isinstance(x280, torch.Tensor):
            print('x280: {}'.format(x280.shape))
        elif isinstance(x280, tuple):
            tuple_shapes = '('
            for item in x280:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x280: {}'.format(tuple_shapes))
        else:
            print('x280: {}'.format(x280))
        x285=torchvision.models.swin_transformer.shifted_window_attention(x273, self.weight22, self.weight23, x280, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias22, proj_bias=self.bias23)
        if x285 is None:
            print('x285: {}'.format(x285))
        elif isinstance(x285, torch.Tensor):
            print('x285: {}'.format(x285.shape))
        elif isinstance(x285, tuple):
            tuple_shapes = '('
            for item in x285:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x285: {}'.format(tuple_shapes))
        else:
            print('x285: {}'.format(x285))
        x286=stochastic_depth(x285, 0.2391304347826087, 'row', False)
        if x286 is None:
            print('x286: {}'.format(x286))
        elif isinstance(x286, torch.Tensor):
            print('x286: {}'.format(x286.shape))
        elif isinstance(x286, tuple):
            tuple_shapes = '('
            for item in x286:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x286: {}'.format(tuple_shapes))
        else:
            print('x286: {}'.format(x286))
        x287=operator.add(x272, x286)
        if x287 is None:
            print('x287: {}'.format(x287))
        elif isinstance(x287, torch.Tensor):
            print('x287: {}'.format(x287.shape))
        elif isinstance(x287, tuple):
            tuple_shapes = '('
            for item in x287:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x287: {}'.format(tuple_shapes))
        else:
            print('x287: {}'.format(x287))
        x288=self.layernorm26(x287)
        if x288 is None:
            print('x288: {}'.format(x288))
        elif isinstance(x288, torch.Tensor):
            print('x288: {}'.format(x288.shape))
        elif isinstance(x288, tuple):
            tuple_shapes = '('
            for item in x288:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x288: {}'.format(tuple_shapes))
        else:
            print('x288: {}'.format(x288))
        x289=self.linear24(x288)
        if x289 is None:
            print('x289: {}'.format(x289))
        elif isinstance(x289, torch.Tensor):
            print('x289: {}'.format(x289.shape))
        elif isinstance(x289, tuple):
            tuple_shapes = '('
            for item in x289:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x289: {}'.format(tuple_shapes))
        else:
            print('x289: {}'.format(x289))
        x290=self.gelu11(x289)
        if x290 is None:
            print('x290: {}'.format(x290))
        elif isinstance(x290, torch.Tensor):
            print('x290: {}'.format(x290.shape))
        elif isinstance(x290, tuple):
            tuple_shapes = '('
            for item in x290:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x290: {}'.format(tuple_shapes))
        else:
            print('x290: {}'.format(x290))
        x291=self.dropout22(x290)
        if x291 is None:
            print('x291: {}'.format(x291))
        elif isinstance(x291, torch.Tensor):
            print('x291: {}'.format(x291.shape))
        elif isinstance(x291, tuple):
            tuple_shapes = '('
            for item in x291:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x291: {}'.format(tuple_shapes))
        else:
            print('x291: {}'.format(x291))
        x292=self.linear25(x291)
        if x292 is None:
            print('x292: {}'.format(x292))
        elif isinstance(x292, torch.Tensor):
            print('x292: {}'.format(x292.shape))
        elif isinstance(x292, tuple):
            tuple_shapes = '('
            for item in x292:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x292: {}'.format(tuple_shapes))
        else:
            print('x292: {}'.format(x292))
        x293=self.dropout23(x292)
        if x293 is None:
            print('x293: {}'.format(x293))
        elif isinstance(x293, torch.Tensor):
            print('x293: {}'.format(x293.shape))
        elif isinstance(x293, tuple):
            tuple_shapes = '('
            for item in x293:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x293: {}'.format(tuple_shapes))
        else:
            print('x293: {}'.format(x293))
        x294=stochastic_depth(x293, 0.2391304347826087, 'row', False)
        if x294 is None:
            print('x294: {}'.format(x294))
        elif isinstance(x294, torch.Tensor):
            print('x294: {}'.format(x294.shape))
        elif isinstance(x294, tuple):
            tuple_shapes = '('
            for item in x294:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x294: {}'.format(tuple_shapes))
        else:
            print('x294: {}'.format(x294))
        x295=operator.add(x287, x294)
        if x295 is None:
            print('x295: {}'.format(x295))
        elif isinstance(x295, torch.Tensor):
            print('x295: {}'.format(x295.shape))
        elif isinstance(x295, tuple):
            tuple_shapes = '('
            for item in x295:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x295: {}'.format(tuple_shapes))
        else:
            print('x295: {}'.format(x295))
        x296=self.layernorm27(x295)
        if x296 is None:
            print('x296: {}'.format(x296))
        elif isinstance(x296, torch.Tensor):
            print('x296: {}'.format(x296.shape))
        elif isinstance(x296, tuple):
            tuple_shapes = '('
            for item in x296:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x296: {}'.format(tuple_shapes))
        else:
            print('x296: {}'.format(x296))
        x299=operator.getitem(self.relative_position_bias_table12, self.relative_position_index12)
        if x299 is None:
            print('x299: {}'.format(x299))
        elif isinstance(x299, torch.Tensor):
            print('x299: {}'.format(x299.shape))
        elif isinstance(x299, tuple):
            tuple_shapes = '('
            for item in x299:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x299: {}'.format(tuple_shapes))
        else:
            print('x299: {}'.format(x299))
        x300=x299.view(49, 49, -1)
        if x300 is None:
            print('x300: {}'.format(x300))
        elif isinstance(x300, torch.Tensor):
            print('x300: {}'.format(x300.shape))
        elif isinstance(x300, tuple):
            tuple_shapes = '('
            for item in x300:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x300: {}'.format(tuple_shapes))
        else:
            print('x300: {}'.format(x300))
        x301=x300.permute(2, 0, 1)
        if x301 is None:
            print('x301: {}'.format(x301))
        elif isinstance(x301, torch.Tensor):
            print('x301: {}'.format(x301.shape))
        elif isinstance(x301, tuple):
            tuple_shapes = '('
            for item in x301:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x301: {}'.format(tuple_shapes))
        else:
            print('x301: {}'.format(x301))
        x302=x301.contiguous()
        if x302 is None:
            print('x302: {}'.format(x302))
        elif isinstance(x302, torch.Tensor):
            print('x302: {}'.format(x302.shape))
        elif isinstance(x302, tuple):
            tuple_shapes = '('
            for item in x302:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x302: {}'.format(tuple_shapes))
        else:
            print('x302: {}'.format(x302))
        x303=x302.unsqueeze(0)
        if x303 is None:
            print('x303: {}'.format(x303))
        elif isinstance(x303, torch.Tensor):
            print('x303: {}'.format(x303.shape))
        elif isinstance(x303, tuple):
            tuple_shapes = '('
            for item in x303:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x303: {}'.format(tuple_shapes))
        else:
            print('x303: {}'.format(x303))
        x308=torchvision.models.swin_transformer.shifted_window_attention(x296, self.weight24, self.weight25, x303, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias24, proj_bias=self.bias25)
        if x308 is None:
            print('x308: {}'.format(x308))
        elif isinstance(x308, torch.Tensor):
            print('x308: {}'.format(x308.shape))
        elif isinstance(x308, tuple):
            tuple_shapes = '('
            for item in x308:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x308: {}'.format(tuple_shapes))
        else:
            print('x308: {}'.format(x308))
        x309=stochastic_depth(x308, 0.2608695652173913, 'row', False)
        if x309 is None:
            print('x309: {}'.format(x309))
        elif isinstance(x309, torch.Tensor):
            print('x309: {}'.format(x309.shape))
        elif isinstance(x309, tuple):
            tuple_shapes = '('
            for item in x309:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x309: {}'.format(tuple_shapes))
        else:
            print('x309: {}'.format(x309))
        x310=operator.add(x295, x309)
        if x310 is None:
            print('x310: {}'.format(x310))
        elif isinstance(x310, torch.Tensor):
            print('x310: {}'.format(x310.shape))
        elif isinstance(x310, tuple):
            tuple_shapes = '('
            for item in x310:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x310: {}'.format(tuple_shapes))
        else:
            print('x310: {}'.format(x310))
        x311=self.layernorm28(x310)
        if x311 is None:
            print('x311: {}'.format(x311))
        elif isinstance(x311, torch.Tensor):
            print('x311: {}'.format(x311.shape))
        elif isinstance(x311, tuple):
            tuple_shapes = '('
            for item in x311:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x311: {}'.format(tuple_shapes))
        else:
            print('x311: {}'.format(x311))
        x312=self.linear26(x311)
        if x312 is None:
            print('x312: {}'.format(x312))
        elif isinstance(x312, torch.Tensor):
            print('x312: {}'.format(x312.shape))
        elif isinstance(x312, tuple):
            tuple_shapes = '('
            for item in x312:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x312: {}'.format(tuple_shapes))
        else:
            print('x312: {}'.format(x312))
        x313=self.gelu12(x312)
        if x313 is None:
            print('x313: {}'.format(x313))
        elif isinstance(x313, torch.Tensor):
            print('x313: {}'.format(x313.shape))
        elif isinstance(x313, tuple):
            tuple_shapes = '('
            for item in x313:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x313: {}'.format(tuple_shapes))
        else:
            print('x313: {}'.format(x313))
        x314=self.dropout24(x313)
        if x314 is None:
            print('x314: {}'.format(x314))
        elif isinstance(x314, torch.Tensor):
            print('x314: {}'.format(x314.shape))
        elif isinstance(x314, tuple):
            tuple_shapes = '('
            for item in x314:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x314: {}'.format(tuple_shapes))
        else:
            print('x314: {}'.format(x314))
        x315=self.linear27(x314)
        if x315 is None:
            print('x315: {}'.format(x315))
        elif isinstance(x315, torch.Tensor):
            print('x315: {}'.format(x315.shape))
        elif isinstance(x315, tuple):
            tuple_shapes = '('
            for item in x315:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x315: {}'.format(tuple_shapes))
        else:
            print('x315: {}'.format(x315))
        x316=self.dropout25(x315)
        if x316 is None:
            print('x316: {}'.format(x316))
        elif isinstance(x316, torch.Tensor):
            print('x316: {}'.format(x316.shape))
        elif isinstance(x316, tuple):
            tuple_shapes = '('
            for item in x316:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x316: {}'.format(tuple_shapes))
        else:
            print('x316: {}'.format(x316))
        x317=stochastic_depth(x316, 0.2608695652173913, 'row', False)
        if x317 is None:
            print('x317: {}'.format(x317))
        elif isinstance(x317, torch.Tensor):
            print('x317: {}'.format(x317.shape))
        elif isinstance(x317, tuple):
            tuple_shapes = '('
            for item in x317:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x317: {}'.format(tuple_shapes))
        else:
            print('x317: {}'.format(x317))
        x318=operator.add(x310, x317)
        if x318 is None:
            print('x318: {}'.format(x318))
        elif isinstance(x318, torch.Tensor):
            print('x318: {}'.format(x318.shape))
        elif isinstance(x318, tuple):
            tuple_shapes = '('
            for item in x318:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x318: {}'.format(tuple_shapes))
        else:
            print('x318: {}'.format(x318))
        x319=self.layernorm29(x318)
        if x319 is None:
            print('x319: {}'.format(x319))
        elif isinstance(x319, torch.Tensor):
            print('x319: {}'.format(x319.shape))
        elif isinstance(x319, tuple):
            tuple_shapes = '('
            for item in x319:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x319: {}'.format(tuple_shapes))
        else:
            print('x319: {}'.format(x319))
        x322=operator.getitem(self.relative_position_bias_table13, self.relative_position_index13)
        if x322 is None:
            print('x322: {}'.format(x322))
        elif isinstance(x322, torch.Tensor):
            print('x322: {}'.format(x322.shape))
        elif isinstance(x322, tuple):
            tuple_shapes = '('
            for item in x322:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x322: {}'.format(tuple_shapes))
        else:
            print('x322: {}'.format(x322))
        x323=x322.view(49, 49, -1)
        if x323 is None:
            print('x323: {}'.format(x323))
        elif isinstance(x323, torch.Tensor):
            print('x323: {}'.format(x323.shape))
        elif isinstance(x323, tuple):
            tuple_shapes = '('
            for item in x323:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x323: {}'.format(tuple_shapes))
        else:
            print('x323: {}'.format(x323))
        x324=x323.permute(2, 0, 1)
        if x324 is None:
            print('x324: {}'.format(x324))
        elif isinstance(x324, torch.Tensor):
            print('x324: {}'.format(x324.shape))
        elif isinstance(x324, tuple):
            tuple_shapes = '('
            for item in x324:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x324: {}'.format(tuple_shapes))
        else:
            print('x324: {}'.format(x324))
        x325=x324.contiguous()
        if x325 is None:
            print('x325: {}'.format(x325))
        elif isinstance(x325, torch.Tensor):
            print('x325: {}'.format(x325.shape))
        elif isinstance(x325, tuple):
            tuple_shapes = '('
            for item in x325:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x325: {}'.format(tuple_shapes))
        else:
            print('x325: {}'.format(x325))
        x326=x325.unsqueeze(0)
        if x326 is None:
            print('x326: {}'.format(x326))
        elif isinstance(x326, torch.Tensor):
            print('x326: {}'.format(x326.shape))
        elif isinstance(x326, tuple):
            tuple_shapes = '('
            for item in x326:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x326: {}'.format(tuple_shapes))
        else:
            print('x326: {}'.format(x326))
        x331=torchvision.models.swin_transformer.shifted_window_attention(x319, self.weight26, self.weight27, x326, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias26, proj_bias=self.bias27)
        if x331 is None:
            print('x331: {}'.format(x331))
        elif isinstance(x331, torch.Tensor):
            print('x331: {}'.format(x331.shape))
        elif isinstance(x331, tuple):
            tuple_shapes = '('
            for item in x331:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x331: {}'.format(tuple_shapes))
        else:
            print('x331: {}'.format(x331))
        x332=stochastic_depth(x331, 0.2826086956521739, 'row', False)
        if x332 is None:
            print('x332: {}'.format(x332))
        elif isinstance(x332, torch.Tensor):
            print('x332: {}'.format(x332.shape))
        elif isinstance(x332, tuple):
            tuple_shapes = '('
            for item in x332:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x332: {}'.format(tuple_shapes))
        else:
            print('x332: {}'.format(x332))
        x333=operator.add(x318, x332)
        if x333 is None:
            print('x333: {}'.format(x333))
        elif isinstance(x333, torch.Tensor):
            print('x333: {}'.format(x333.shape))
        elif isinstance(x333, tuple):
            tuple_shapes = '('
            for item in x333:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x333: {}'.format(tuple_shapes))
        else:
            print('x333: {}'.format(x333))
        x334=self.layernorm30(x333)
        if x334 is None:
            print('x334: {}'.format(x334))
        elif isinstance(x334, torch.Tensor):
            print('x334: {}'.format(x334.shape))
        elif isinstance(x334, tuple):
            tuple_shapes = '('
            for item in x334:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x334: {}'.format(tuple_shapes))
        else:
            print('x334: {}'.format(x334))
        x335=self.linear28(x334)
        if x335 is None:
            print('x335: {}'.format(x335))
        elif isinstance(x335, torch.Tensor):
            print('x335: {}'.format(x335.shape))
        elif isinstance(x335, tuple):
            tuple_shapes = '('
            for item in x335:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x335: {}'.format(tuple_shapes))
        else:
            print('x335: {}'.format(x335))
        x336=self.gelu13(x335)
        if x336 is None:
            print('x336: {}'.format(x336))
        elif isinstance(x336, torch.Tensor):
            print('x336: {}'.format(x336.shape))
        elif isinstance(x336, tuple):
            tuple_shapes = '('
            for item in x336:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x336: {}'.format(tuple_shapes))
        else:
            print('x336: {}'.format(x336))
        x337=self.dropout26(x336)
        if x337 is None:
            print('x337: {}'.format(x337))
        elif isinstance(x337, torch.Tensor):
            print('x337: {}'.format(x337.shape))
        elif isinstance(x337, tuple):
            tuple_shapes = '('
            for item in x337:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x337: {}'.format(tuple_shapes))
        else:
            print('x337: {}'.format(x337))
        x338=self.linear29(x337)
        if x338 is None:
            print('x338: {}'.format(x338))
        elif isinstance(x338, torch.Tensor):
            print('x338: {}'.format(x338.shape))
        elif isinstance(x338, tuple):
            tuple_shapes = '('
            for item in x338:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x338: {}'.format(tuple_shapes))
        else:
            print('x338: {}'.format(x338))
        x339=self.dropout27(x338)
        if x339 is None:
            print('x339: {}'.format(x339))
        elif isinstance(x339, torch.Tensor):
            print('x339: {}'.format(x339.shape))
        elif isinstance(x339, tuple):
            tuple_shapes = '('
            for item in x339:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x339: {}'.format(tuple_shapes))
        else:
            print('x339: {}'.format(x339))
        x340=stochastic_depth(x339, 0.2826086956521739, 'row', False)
        if x340 is None:
            print('x340: {}'.format(x340))
        elif isinstance(x340, torch.Tensor):
            print('x340: {}'.format(x340.shape))
        elif isinstance(x340, tuple):
            tuple_shapes = '('
            for item in x340:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x340: {}'.format(tuple_shapes))
        else:
            print('x340: {}'.format(x340))
        x341=operator.add(x333, x340)
        if x341 is None:
            print('x341: {}'.format(x341))
        elif isinstance(x341, torch.Tensor):
            print('x341: {}'.format(x341.shape))
        elif isinstance(x341, tuple):
            tuple_shapes = '('
            for item in x341:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x341: {}'.format(tuple_shapes))
        else:
            print('x341: {}'.format(x341))
        x342=self.layernorm31(x341)
        if x342 is None:
            print('x342: {}'.format(x342))
        elif isinstance(x342, torch.Tensor):
            print('x342: {}'.format(x342.shape))
        elif isinstance(x342, tuple):
            tuple_shapes = '('
            for item in x342:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x342: {}'.format(tuple_shapes))
        else:
            print('x342: {}'.format(x342))
        x345=operator.getitem(self.relative_position_bias_table14, self.relative_position_index14)
        if x345 is None:
            print('x345: {}'.format(x345))
        elif isinstance(x345, torch.Tensor):
            print('x345: {}'.format(x345.shape))
        elif isinstance(x345, tuple):
            tuple_shapes = '('
            for item in x345:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x345: {}'.format(tuple_shapes))
        else:
            print('x345: {}'.format(x345))
        x346=x345.view(49, 49, -1)
        if x346 is None:
            print('x346: {}'.format(x346))
        elif isinstance(x346, torch.Tensor):
            print('x346: {}'.format(x346.shape))
        elif isinstance(x346, tuple):
            tuple_shapes = '('
            for item in x346:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x346: {}'.format(tuple_shapes))
        else:
            print('x346: {}'.format(x346))
        x347=x346.permute(2, 0, 1)
        if x347 is None:
            print('x347: {}'.format(x347))
        elif isinstance(x347, torch.Tensor):
            print('x347: {}'.format(x347.shape))
        elif isinstance(x347, tuple):
            tuple_shapes = '('
            for item in x347:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x347: {}'.format(tuple_shapes))
        else:
            print('x347: {}'.format(x347))
        x348=x347.contiguous()
        if x348 is None:
            print('x348: {}'.format(x348))
        elif isinstance(x348, torch.Tensor):
            print('x348: {}'.format(x348.shape))
        elif isinstance(x348, tuple):
            tuple_shapes = '('
            for item in x348:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x348: {}'.format(tuple_shapes))
        else:
            print('x348: {}'.format(x348))
        x349=x348.unsqueeze(0)
        if x349 is None:
            print('x349: {}'.format(x349))
        elif isinstance(x349, torch.Tensor):
            print('x349: {}'.format(x349.shape))
        elif isinstance(x349, tuple):
            tuple_shapes = '('
            for item in x349:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x349: {}'.format(tuple_shapes))
        else:
            print('x349: {}'.format(x349))
        x354=torchvision.models.swin_transformer.shifted_window_attention(x342, self.weight28, self.weight29, x349, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias28, proj_bias=self.bias29)
        if x354 is None:
            print('x354: {}'.format(x354))
        elif isinstance(x354, torch.Tensor):
            print('x354: {}'.format(x354.shape))
        elif isinstance(x354, tuple):
            tuple_shapes = '('
            for item in x354:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x354: {}'.format(tuple_shapes))
        else:
            print('x354: {}'.format(x354))
        x355=stochastic_depth(x354, 0.30434782608695654, 'row', False)
        if x355 is None:
            print('x355: {}'.format(x355))
        elif isinstance(x355, torch.Tensor):
            print('x355: {}'.format(x355.shape))
        elif isinstance(x355, tuple):
            tuple_shapes = '('
            for item in x355:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x355: {}'.format(tuple_shapes))
        else:
            print('x355: {}'.format(x355))
        x356=operator.add(x341, x355)
        if x356 is None:
            print('x356: {}'.format(x356))
        elif isinstance(x356, torch.Tensor):
            print('x356: {}'.format(x356.shape))
        elif isinstance(x356, tuple):
            tuple_shapes = '('
            for item in x356:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x356: {}'.format(tuple_shapes))
        else:
            print('x356: {}'.format(x356))
        x357=self.layernorm32(x356)
        if x357 is None:
            print('x357: {}'.format(x357))
        elif isinstance(x357, torch.Tensor):
            print('x357: {}'.format(x357.shape))
        elif isinstance(x357, tuple):
            tuple_shapes = '('
            for item in x357:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x357: {}'.format(tuple_shapes))
        else:
            print('x357: {}'.format(x357))
        x358=self.linear30(x357)
        if x358 is None:
            print('x358: {}'.format(x358))
        elif isinstance(x358, torch.Tensor):
            print('x358: {}'.format(x358.shape))
        elif isinstance(x358, tuple):
            tuple_shapes = '('
            for item in x358:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x358: {}'.format(tuple_shapes))
        else:
            print('x358: {}'.format(x358))
        x359=self.gelu14(x358)
        if x359 is None:
            print('x359: {}'.format(x359))
        elif isinstance(x359, torch.Tensor):
            print('x359: {}'.format(x359.shape))
        elif isinstance(x359, tuple):
            tuple_shapes = '('
            for item in x359:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x359: {}'.format(tuple_shapes))
        else:
            print('x359: {}'.format(x359))
        x360=self.dropout28(x359)
        if x360 is None:
            print('x360: {}'.format(x360))
        elif isinstance(x360, torch.Tensor):
            print('x360: {}'.format(x360.shape))
        elif isinstance(x360, tuple):
            tuple_shapes = '('
            for item in x360:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x360: {}'.format(tuple_shapes))
        else:
            print('x360: {}'.format(x360))
        x361=self.linear31(x360)
        if x361 is None:
            print('x361: {}'.format(x361))
        elif isinstance(x361, torch.Tensor):
            print('x361: {}'.format(x361.shape))
        elif isinstance(x361, tuple):
            tuple_shapes = '('
            for item in x361:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x361: {}'.format(tuple_shapes))
        else:
            print('x361: {}'.format(x361))
        x362=self.dropout29(x361)
        if x362 is None:
            print('x362: {}'.format(x362))
        elif isinstance(x362, torch.Tensor):
            print('x362: {}'.format(x362.shape))
        elif isinstance(x362, tuple):
            tuple_shapes = '('
            for item in x362:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x362: {}'.format(tuple_shapes))
        else:
            print('x362: {}'.format(x362))
        x363=stochastic_depth(x362, 0.30434782608695654, 'row', False)
        if x363 is None:
            print('x363: {}'.format(x363))
        elif isinstance(x363, torch.Tensor):
            print('x363: {}'.format(x363.shape))
        elif isinstance(x363, tuple):
            tuple_shapes = '('
            for item in x363:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x363: {}'.format(tuple_shapes))
        else:
            print('x363: {}'.format(x363))
        x364=operator.add(x356, x363)
        if x364 is None:
            print('x364: {}'.format(x364))
        elif isinstance(x364, torch.Tensor):
            print('x364: {}'.format(x364.shape))
        elif isinstance(x364, tuple):
            tuple_shapes = '('
            for item in x364:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x364: {}'.format(tuple_shapes))
        else:
            print('x364: {}'.format(x364))
        x365=self.layernorm33(x364)
        if x365 is None:
            print('x365: {}'.format(x365))
        elif isinstance(x365, torch.Tensor):
            print('x365: {}'.format(x365.shape))
        elif isinstance(x365, tuple):
            tuple_shapes = '('
            for item in x365:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x365: {}'.format(tuple_shapes))
        else:
            print('x365: {}'.format(x365))
        x368=operator.getitem(self.relative_position_bias_table15, self.relative_position_index15)
        if x368 is None:
            print('x368: {}'.format(x368))
        elif isinstance(x368, torch.Tensor):
            print('x368: {}'.format(x368.shape))
        elif isinstance(x368, tuple):
            tuple_shapes = '('
            for item in x368:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x368: {}'.format(tuple_shapes))
        else:
            print('x368: {}'.format(x368))
        x369=x368.view(49, 49, -1)
        if x369 is None:
            print('x369: {}'.format(x369))
        elif isinstance(x369, torch.Tensor):
            print('x369: {}'.format(x369.shape))
        elif isinstance(x369, tuple):
            tuple_shapes = '('
            for item in x369:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x369: {}'.format(tuple_shapes))
        else:
            print('x369: {}'.format(x369))
        x370=x369.permute(2, 0, 1)
        if x370 is None:
            print('x370: {}'.format(x370))
        elif isinstance(x370, torch.Tensor):
            print('x370: {}'.format(x370.shape))
        elif isinstance(x370, tuple):
            tuple_shapes = '('
            for item in x370:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x370: {}'.format(tuple_shapes))
        else:
            print('x370: {}'.format(x370))
        x371=x370.contiguous()
        if x371 is None:
            print('x371: {}'.format(x371))
        elif isinstance(x371, torch.Tensor):
            print('x371: {}'.format(x371.shape))
        elif isinstance(x371, tuple):
            tuple_shapes = '('
            for item in x371:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x371: {}'.format(tuple_shapes))
        else:
            print('x371: {}'.format(x371))
        x372=x371.unsqueeze(0)
        if x372 is None:
            print('x372: {}'.format(x372))
        elif isinstance(x372, torch.Tensor):
            print('x372: {}'.format(x372.shape))
        elif isinstance(x372, tuple):
            tuple_shapes = '('
            for item in x372:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x372: {}'.format(tuple_shapes))
        else:
            print('x372: {}'.format(x372))
        x377=torchvision.models.swin_transformer.shifted_window_attention(x365, self.weight30, self.weight31, x372, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias30, proj_bias=self.bias31)
        if x377 is None:
            print('x377: {}'.format(x377))
        elif isinstance(x377, torch.Tensor):
            print('x377: {}'.format(x377.shape))
        elif isinstance(x377, tuple):
            tuple_shapes = '('
            for item in x377:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x377: {}'.format(tuple_shapes))
        else:
            print('x377: {}'.format(x377))
        x378=stochastic_depth(x377, 0.32608695652173914, 'row', False)
        if x378 is None:
            print('x378: {}'.format(x378))
        elif isinstance(x378, torch.Tensor):
            print('x378: {}'.format(x378.shape))
        elif isinstance(x378, tuple):
            tuple_shapes = '('
            for item in x378:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x378: {}'.format(tuple_shapes))
        else:
            print('x378: {}'.format(x378))
        x379=operator.add(x364, x378)
        if x379 is None:
            print('x379: {}'.format(x379))
        elif isinstance(x379, torch.Tensor):
            print('x379: {}'.format(x379.shape))
        elif isinstance(x379, tuple):
            tuple_shapes = '('
            for item in x379:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x379: {}'.format(tuple_shapes))
        else:
            print('x379: {}'.format(x379))
        x380=self.layernorm34(x379)
        if x380 is None:
            print('x380: {}'.format(x380))
        elif isinstance(x380, torch.Tensor):
            print('x380: {}'.format(x380.shape))
        elif isinstance(x380, tuple):
            tuple_shapes = '('
            for item in x380:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x380: {}'.format(tuple_shapes))
        else:
            print('x380: {}'.format(x380))
        x381=self.linear32(x380)
        if x381 is None:
            print('x381: {}'.format(x381))
        elif isinstance(x381, torch.Tensor):
            print('x381: {}'.format(x381.shape))
        elif isinstance(x381, tuple):
            tuple_shapes = '('
            for item in x381:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x381: {}'.format(tuple_shapes))
        else:
            print('x381: {}'.format(x381))
        x382=self.gelu15(x381)
        if x382 is None:
            print('x382: {}'.format(x382))
        elif isinstance(x382, torch.Tensor):
            print('x382: {}'.format(x382.shape))
        elif isinstance(x382, tuple):
            tuple_shapes = '('
            for item in x382:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x382: {}'.format(tuple_shapes))
        else:
            print('x382: {}'.format(x382))
        x383=self.dropout30(x382)
        if x383 is None:
            print('x383: {}'.format(x383))
        elif isinstance(x383, torch.Tensor):
            print('x383: {}'.format(x383.shape))
        elif isinstance(x383, tuple):
            tuple_shapes = '('
            for item in x383:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x383: {}'.format(tuple_shapes))
        else:
            print('x383: {}'.format(x383))
        x384=self.linear33(x383)
        if x384 is None:
            print('x384: {}'.format(x384))
        elif isinstance(x384, torch.Tensor):
            print('x384: {}'.format(x384.shape))
        elif isinstance(x384, tuple):
            tuple_shapes = '('
            for item in x384:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x384: {}'.format(tuple_shapes))
        else:
            print('x384: {}'.format(x384))
        x385=self.dropout31(x384)
        if x385 is None:
            print('x385: {}'.format(x385))
        elif isinstance(x385, torch.Tensor):
            print('x385: {}'.format(x385.shape))
        elif isinstance(x385, tuple):
            tuple_shapes = '('
            for item in x385:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x385: {}'.format(tuple_shapes))
        else:
            print('x385: {}'.format(x385))
        x386=stochastic_depth(x385, 0.32608695652173914, 'row', False)
        if x386 is None:
            print('x386: {}'.format(x386))
        elif isinstance(x386, torch.Tensor):
            print('x386: {}'.format(x386.shape))
        elif isinstance(x386, tuple):
            tuple_shapes = '('
            for item in x386:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x386: {}'.format(tuple_shapes))
        else:
            print('x386: {}'.format(x386))
        x387=operator.add(x379, x386)
        if x387 is None:
            print('x387: {}'.format(x387))
        elif isinstance(x387, torch.Tensor):
            print('x387: {}'.format(x387.shape))
        elif isinstance(x387, tuple):
            tuple_shapes = '('
            for item in x387:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x387: {}'.format(tuple_shapes))
        else:
            print('x387: {}'.format(x387))
        x388=self.layernorm35(x387)
        if x388 is None:
            print('x388: {}'.format(x388))
        elif isinstance(x388, torch.Tensor):
            print('x388: {}'.format(x388.shape))
        elif isinstance(x388, tuple):
            tuple_shapes = '('
            for item in x388:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x388: {}'.format(tuple_shapes))
        else:
            print('x388: {}'.format(x388))
        x391=operator.getitem(self.relative_position_bias_table16, self.relative_position_index16)
        if x391 is None:
            print('x391: {}'.format(x391))
        elif isinstance(x391, torch.Tensor):
            print('x391: {}'.format(x391.shape))
        elif isinstance(x391, tuple):
            tuple_shapes = '('
            for item in x391:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x391: {}'.format(tuple_shapes))
        else:
            print('x391: {}'.format(x391))
        x392=x391.view(49, 49, -1)
        if x392 is None:
            print('x392: {}'.format(x392))
        elif isinstance(x392, torch.Tensor):
            print('x392: {}'.format(x392.shape))
        elif isinstance(x392, tuple):
            tuple_shapes = '('
            for item in x392:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x392: {}'.format(tuple_shapes))
        else:
            print('x392: {}'.format(x392))
        x393=x392.permute(2, 0, 1)
        if x393 is None:
            print('x393: {}'.format(x393))
        elif isinstance(x393, torch.Tensor):
            print('x393: {}'.format(x393.shape))
        elif isinstance(x393, tuple):
            tuple_shapes = '('
            for item in x393:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x393: {}'.format(tuple_shapes))
        else:
            print('x393: {}'.format(x393))
        x394=x393.contiguous()
        if x394 is None:
            print('x394: {}'.format(x394))
        elif isinstance(x394, torch.Tensor):
            print('x394: {}'.format(x394.shape))
        elif isinstance(x394, tuple):
            tuple_shapes = '('
            for item in x394:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x394: {}'.format(tuple_shapes))
        else:
            print('x394: {}'.format(x394))
        x395=x394.unsqueeze(0)
        if x395 is None:
            print('x395: {}'.format(x395))
        elif isinstance(x395, torch.Tensor):
            print('x395: {}'.format(x395.shape))
        elif isinstance(x395, tuple):
            tuple_shapes = '('
            for item in x395:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x395: {}'.format(tuple_shapes))
        else:
            print('x395: {}'.format(x395))
        x400=torchvision.models.swin_transformer.shifted_window_attention(x388, self.weight32, self.weight33, x395, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias32, proj_bias=self.bias33)
        if x400 is None:
            print('x400: {}'.format(x400))
        elif isinstance(x400, torch.Tensor):
            print('x400: {}'.format(x400.shape))
        elif isinstance(x400, tuple):
            tuple_shapes = '('
            for item in x400:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x400: {}'.format(tuple_shapes))
        else:
            print('x400: {}'.format(x400))
        x401=stochastic_depth(x400, 0.34782608695652173, 'row', False)
        if x401 is None:
            print('x401: {}'.format(x401))
        elif isinstance(x401, torch.Tensor):
            print('x401: {}'.format(x401.shape))
        elif isinstance(x401, tuple):
            tuple_shapes = '('
            for item in x401:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x401: {}'.format(tuple_shapes))
        else:
            print('x401: {}'.format(x401))
        x402=operator.add(x387, x401)
        if x402 is None:
            print('x402: {}'.format(x402))
        elif isinstance(x402, torch.Tensor):
            print('x402: {}'.format(x402.shape))
        elif isinstance(x402, tuple):
            tuple_shapes = '('
            for item in x402:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x402: {}'.format(tuple_shapes))
        else:
            print('x402: {}'.format(x402))
        x403=self.layernorm36(x402)
        if x403 is None:
            print('x403: {}'.format(x403))
        elif isinstance(x403, torch.Tensor):
            print('x403: {}'.format(x403.shape))
        elif isinstance(x403, tuple):
            tuple_shapes = '('
            for item in x403:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x403: {}'.format(tuple_shapes))
        else:
            print('x403: {}'.format(x403))
        x404=self.linear34(x403)
        if x404 is None:
            print('x404: {}'.format(x404))
        elif isinstance(x404, torch.Tensor):
            print('x404: {}'.format(x404.shape))
        elif isinstance(x404, tuple):
            tuple_shapes = '('
            for item in x404:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x404: {}'.format(tuple_shapes))
        else:
            print('x404: {}'.format(x404))
        x405=self.gelu16(x404)
        if x405 is None:
            print('x405: {}'.format(x405))
        elif isinstance(x405, torch.Tensor):
            print('x405: {}'.format(x405.shape))
        elif isinstance(x405, tuple):
            tuple_shapes = '('
            for item in x405:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x405: {}'.format(tuple_shapes))
        else:
            print('x405: {}'.format(x405))
        x406=self.dropout32(x405)
        if x406 is None:
            print('x406: {}'.format(x406))
        elif isinstance(x406, torch.Tensor):
            print('x406: {}'.format(x406.shape))
        elif isinstance(x406, tuple):
            tuple_shapes = '('
            for item in x406:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x406: {}'.format(tuple_shapes))
        else:
            print('x406: {}'.format(x406))
        x407=self.linear35(x406)
        if x407 is None:
            print('x407: {}'.format(x407))
        elif isinstance(x407, torch.Tensor):
            print('x407: {}'.format(x407.shape))
        elif isinstance(x407, tuple):
            tuple_shapes = '('
            for item in x407:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x407: {}'.format(tuple_shapes))
        else:
            print('x407: {}'.format(x407))
        x408=self.dropout33(x407)
        if x408 is None:
            print('x408: {}'.format(x408))
        elif isinstance(x408, torch.Tensor):
            print('x408: {}'.format(x408.shape))
        elif isinstance(x408, tuple):
            tuple_shapes = '('
            for item in x408:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x408: {}'.format(tuple_shapes))
        else:
            print('x408: {}'.format(x408))
        x409=stochastic_depth(x408, 0.34782608695652173, 'row', False)
        if x409 is None:
            print('x409: {}'.format(x409))
        elif isinstance(x409, torch.Tensor):
            print('x409: {}'.format(x409.shape))
        elif isinstance(x409, tuple):
            tuple_shapes = '('
            for item in x409:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x409: {}'.format(tuple_shapes))
        else:
            print('x409: {}'.format(x409))
        x410=operator.add(x402, x409)
        if x410 is None:
            print('x410: {}'.format(x410))
        elif isinstance(x410, torch.Tensor):
            print('x410: {}'.format(x410.shape))
        elif isinstance(x410, tuple):
            tuple_shapes = '('
            for item in x410:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x410: {}'.format(tuple_shapes))
        else:
            print('x410: {}'.format(x410))
        x411=self.layernorm37(x410)
        if x411 is None:
            print('x411: {}'.format(x411))
        elif isinstance(x411, torch.Tensor):
            print('x411: {}'.format(x411.shape))
        elif isinstance(x411, tuple):
            tuple_shapes = '('
            for item in x411:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x411: {}'.format(tuple_shapes))
        else:
            print('x411: {}'.format(x411))
        x414=operator.getitem(self.relative_position_bias_table17, self.relative_position_index17)
        if x414 is None:
            print('x414: {}'.format(x414))
        elif isinstance(x414, torch.Tensor):
            print('x414: {}'.format(x414.shape))
        elif isinstance(x414, tuple):
            tuple_shapes = '('
            for item in x414:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x414: {}'.format(tuple_shapes))
        else:
            print('x414: {}'.format(x414))
        x415=x414.view(49, 49, -1)
        if x415 is None:
            print('x415: {}'.format(x415))
        elif isinstance(x415, torch.Tensor):
            print('x415: {}'.format(x415.shape))
        elif isinstance(x415, tuple):
            tuple_shapes = '('
            for item in x415:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x415: {}'.format(tuple_shapes))
        else:
            print('x415: {}'.format(x415))
        x416=x415.permute(2, 0, 1)
        if x416 is None:
            print('x416: {}'.format(x416))
        elif isinstance(x416, torch.Tensor):
            print('x416: {}'.format(x416.shape))
        elif isinstance(x416, tuple):
            tuple_shapes = '('
            for item in x416:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x416: {}'.format(tuple_shapes))
        else:
            print('x416: {}'.format(x416))
        x417=x416.contiguous()
        if x417 is None:
            print('x417: {}'.format(x417))
        elif isinstance(x417, torch.Tensor):
            print('x417: {}'.format(x417.shape))
        elif isinstance(x417, tuple):
            tuple_shapes = '('
            for item in x417:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x417: {}'.format(tuple_shapes))
        else:
            print('x417: {}'.format(x417))
        x418=x417.unsqueeze(0)
        if x418 is None:
            print('x418: {}'.format(x418))
        elif isinstance(x418, torch.Tensor):
            print('x418: {}'.format(x418.shape))
        elif isinstance(x418, tuple):
            tuple_shapes = '('
            for item in x418:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x418: {}'.format(tuple_shapes))
        else:
            print('x418: {}'.format(x418))
        x423=torchvision.models.swin_transformer.shifted_window_attention(x411, self.weight34, self.weight35, x418, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias34, proj_bias=self.bias35)
        if x423 is None:
            print('x423: {}'.format(x423))
        elif isinstance(x423, torch.Tensor):
            print('x423: {}'.format(x423.shape))
        elif isinstance(x423, tuple):
            tuple_shapes = '('
            for item in x423:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x423: {}'.format(tuple_shapes))
        else:
            print('x423: {}'.format(x423))
        x424=stochastic_depth(x423, 0.3695652173913043, 'row', False)
        if x424 is None:
            print('x424: {}'.format(x424))
        elif isinstance(x424, torch.Tensor):
            print('x424: {}'.format(x424.shape))
        elif isinstance(x424, tuple):
            tuple_shapes = '('
            for item in x424:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x424: {}'.format(tuple_shapes))
        else:
            print('x424: {}'.format(x424))
        x425=operator.add(x410, x424)
        if x425 is None:
            print('x425: {}'.format(x425))
        elif isinstance(x425, torch.Tensor):
            print('x425: {}'.format(x425.shape))
        elif isinstance(x425, tuple):
            tuple_shapes = '('
            for item in x425:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x425: {}'.format(tuple_shapes))
        else:
            print('x425: {}'.format(x425))
        x426=self.layernorm38(x425)
        if x426 is None:
            print('x426: {}'.format(x426))
        elif isinstance(x426, torch.Tensor):
            print('x426: {}'.format(x426.shape))
        elif isinstance(x426, tuple):
            tuple_shapes = '('
            for item in x426:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x426: {}'.format(tuple_shapes))
        else:
            print('x426: {}'.format(x426))
        x427=self.linear36(x426)
        if x427 is None:
            print('x427: {}'.format(x427))
        elif isinstance(x427, torch.Tensor):
            print('x427: {}'.format(x427.shape))
        elif isinstance(x427, tuple):
            tuple_shapes = '('
            for item in x427:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x427: {}'.format(tuple_shapes))
        else:
            print('x427: {}'.format(x427))
        x428=self.gelu17(x427)
        if x428 is None:
            print('x428: {}'.format(x428))
        elif isinstance(x428, torch.Tensor):
            print('x428: {}'.format(x428.shape))
        elif isinstance(x428, tuple):
            tuple_shapes = '('
            for item in x428:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x428: {}'.format(tuple_shapes))
        else:
            print('x428: {}'.format(x428))
        x429=self.dropout34(x428)
        if x429 is None:
            print('x429: {}'.format(x429))
        elif isinstance(x429, torch.Tensor):
            print('x429: {}'.format(x429.shape))
        elif isinstance(x429, tuple):
            tuple_shapes = '('
            for item in x429:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x429: {}'.format(tuple_shapes))
        else:
            print('x429: {}'.format(x429))
        x430=self.linear37(x429)
        if x430 is None:
            print('x430: {}'.format(x430))
        elif isinstance(x430, torch.Tensor):
            print('x430: {}'.format(x430.shape))
        elif isinstance(x430, tuple):
            tuple_shapes = '('
            for item in x430:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x430: {}'.format(tuple_shapes))
        else:
            print('x430: {}'.format(x430))
        x431=self.dropout35(x430)
        if x431 is None:
            print('x431: {}'.format(x431))
        elif isinstance(x431, torch.Tensor):
            print('x431: {}'.format(x431.shape))
        elif isinstance(x431, tuple):
            tuple_shapes = '('
            for item in x431:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x431: {}'.format(tuple_shapes))
        else:
            print('x431: {}'.format(x431))
        x432=stochastic_depth(x431, 0.3695652173913043, 'row', False)
        if x432 is None:
            print('x432: {}'.format(x432))
        elif isinstance(x432, torch.Tensor):
            print('x432: {}'.format(x432.shape))
        elif isinstance(x432, tuple):
            tuple_shapes = '('
            for item in x432:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x432: {}'.format(tuple_shapes))
        else:
            print('x432: {}'.format(x432))
        x433=operator.add(x425, x432)
        if x433 is None:
            print('x433: {}'.format(x433))
        elif isinstance(x433, torch.Tensor):
            print('x433: {}'.format(x433.shape))
        elif isinstance(x433, tuple):
            tuple_shapes = '('
            for item in x433:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x433: {}'.format(tuple_shapes))
        else:
            print('x433: {}'.format(x433))
        x434=self.layernorm39(x433)
        if x434 is None:
            print('x434: {}'.format(x434))
        elif isinstance(x434, torch.Tensor):
            print('x434: {}'.format(x434.shape))
        elif isinstance(x434, tuple):
            tuple_shapes = '('
            for item in x434:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x434: {}'.format(tuple_shapes))
        else:
            print('x434: {}'.format(x434))
        x437=operator.getitem(self.relative_position_bias_table18, self.relative_position_index18)
        if x437 is None:
            print('x437: {}'.format(x437))
        elif isinstance(x437, torch.Tensor):
            print('x437: {}'.format(x437.shape))
        elif isinstance(x437, tuple):
            tuple_shapes = '('
            for item in x437:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x437: {}'.format(tuple_shapes))
        else:
            print('x437: {}'.format(x437))
        x438=x437.view(49, 49, -1)
        if x438 is None:
            print('x438: {}'.format(x438))
        elif isinstance(x438, torch.Tensor):
            print('x438: {}'.format(x438.shape))
        elif isinstance(x438, tuple):
            tuple_shapes = '('
            for item in x438:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x438: {}'.format(tuple_shapes))
        else:
            print('x438: {}'.format(x438))
        x439=x438.permute(2, 0, 1)
        if x439 is None:
            print('x439: {}'.format(x439))
        elif isinstance(x439, torch.Tensor):
            print('x439: {}'.format(x439.shape))
        elif isinstance(x439, tuple):
            tuple_shapes = '('
            for item in x439:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x439: {}'.format(tuple_shapes))
        else:
            print('x439: {}'.format(x439))
        x440=x439.contiguous()
        if x440 is None:
            print('x440: {}'.format(x440))
        elif isinstance(x440, torch.Tensor):
            print('x440: {}'.format(x440.shape))
        elif isinstance(x440, tuple):
            tuple_shapes = '('
            for item in x440:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x440: {}'.format(tuple_shapes))
        else:
            print('x440: {}'.format(x440))
        x441=x440.unsqueeze(0)
        if x441 is None:
            print('x441: {}'.format(x441))
        elif isinstance(x441, torch.Tensor):
            print('x441: {}'.format(x441.shape))
        elif isinstance(x441, tuple):
            tuple_shapes = '('
            for item in x441:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x441: {}'.format(tuple_shapes))
        else:
            print('x441: {}'.format(x441))
        x446=torchvision.models.swin_transformer.shifted_window_attention(x434, self.weight36, self.weight37, x441, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias36, proj_bias=self.bias37)
        if x446 is None:
            print('x446: {}'.format(x446))
        elif isinstance(x446, torch.Tensor):
            print('x446: {}'.format(x446.shape))
        elif isinstance(x446, tuple):
            tuple_shapes = '('
            for item in x446:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x446: {}'.format(tuple_shapes))
        else:
            print('x446: {}'.format(x446))
        x447=stochastic_depth(x446, 0.391304347826087, 'row', False)
        if x447 is None:
            print('x447: {}'.format(x447))
        elif isinstance(x447, torch.Tensor):
            print('x447: {}'.format(x447.shape))
        elif isinstance(x447, tuple):
            tuple_shapes = '('
            for item in x447:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x447: {}'.format(tuple_shapes))
        else:
            print('x447: {}'.format(x447))
        x448=operator.add(x433, x447)
        if x448 is None:
            print('x448: {}'.format(x448))
        elif isinstance(x448, torch.Tensor):
            print('x448: {}'.format(x448.shape))
        elif isinstance(x448, tuple):
            tuple_shapes = '('
            for item in x448:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x448: {}'.format(tuple_shapes))
        else:
            print('x448: {}'.format(x448))
        x449=self.layernorm40(x448)
        if x449 is None:
            print('x449: {}'.format(x449))
        elif isinstance(x449, torch.Tensor):
            print('x449: {}'.format(x449.shape))
        elif isinstance(x449, tuple):
            tuple_shapes = '('
            for item in x449:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x449: {}'.format(tuple_shapes))
        else:
            print('x449: {}'.format(x449))
        x450=self.linear38(x449)
        if x450 is None:
            print('x450: {}'.format(x450))
        elif isinstance(x450, torch.Tensor):
            print('x450: {}'.format(x450.shape))
        elif isinstance(x450, tuple):
            tuple_shapes = '('
            for item in x450:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x450: {}'.format(tuple_shapes))
        else:
            print('x450: {}'.format(x450))
        x451=self.gelu18(x450)
        if x451 is None:
            print('x451: {}'.format(x451))
        elif isinstance(x451, torch.Tensor):
            print('x451: {}'.format(x451.shape))
        elif isinstance(x451, tuple):
            tuple_shapes = '('
            for item in x451:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x451: {}'.format(tuple_shapes))
        else:
            print('x451: {}'.format(x451))
        x452=self.dropout36(x451)
        if x452 is None:
            print('x452: {}'.format(x452))
        elif isinstance(x452, torch.Tensor):
            print('x452: {}'.format(x452.shape))
        elif isinstance(x452, tuple):
            tuple_shapes = '('
            for item in x452:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x452: {}'.format(tuple_shapes))
        else:
            print('x452: {}'.format(x452))
        x453=self.linear39(x452)
        if x453 is None:
            print('x453: {}'.format(x453))
        elif isinstance(x453, torch.Tensor):
            print('x453: {}'.format(x453.shape))
        elif isinstance(x453, tuple):
            tuple_shapes = '('
            for item in x453:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x453: {}'.format(tuple_shapes))
        else:
            print('x453: {}'.format(x453))
        x454=self.dropout37(x453)
        if x454 is None:
            print('x454: {}'.format(x454))
        elif isinstance(x454, torch.Tensor):
            print('x454: {}'.format(x454.shape))
        elif isinstance(x454, tuple):
            tuple_shapes = '('
            for item in x454:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x454: {}'.format(tuple_shapes))
        else:
            print('x454: {}'.format(x454))
        x455=stochastic_depth(x454, 0.391304347826087, 'row', False)
        if x455 is None:
            print('x455: {}'.format(x455))
        elif isinstance(x455, torch.Tensor):
            print('x455: {}'.format(x455.shape))
        elif isinstance(x455, tuple):
            tuple_shapes = '('
            for item in x455:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x455: {}'.format(tuple_shapes))
        else:
            print('x455: {}'.format(x455))
        x456=operator.add(x448, x455)
        if x456 is None:
            print('x456: {}'.format(x456))
        elif isinstance(x456, torch.Tensor):
            print('x456: {}'.format(x456.shape))
        elif isinstance(x456, tuple):
            tuple_shapes = '('
            for item in x456:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x456: {}'.format(tuple_shapes))
        else:
            print('x456: {}'.format(x456))
        x457=self.layernorm41(x456)
        if x457 is None:
            print('x457: {}'.format(x457))
        elif isinstance(x457, torch.Tensor):
            print('x457: {}'.format(x457.shape))
        elif isinstance(x457, tuple):
            tuple_shapes = '('
            for item in x457:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x457: {}'.format(tuple_shapes))
        else:
            print('x457: {}'.format(x457))
        x460=operator.getitem(self.relative_position_bias_table19, self.relative_position_index19)
        if x460 is None:
            print('x460: {}'.format(x460))
        elif isinstance(x460, torch.Tensor):
            print('x460: {}'.format(x460.shape))
        elif isinstance(x460, tuple):
            tuple_shapes = '('
            for item in x460:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x460: {}'.format(tuple_shapes))
        else:
            print('x460: {}'.format(x460))
        x461=x460.view(49, 49, -1)
        if x461 is None:
            print('x461: {}'.format(x461))
        elif isinstance(x461, torch.Tensor):
            print('x461: {}'.format(x461.shape))
        elif isinstance(x461, tuple):
            tuple_shapes = '('
            for item in x461:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x461: {}'.format(tuple_shapes))
        else:
            print('x461: {}'.format(x461))
        x462=x461.permute(2, 0, 1)
        if x462 is None:
            print('x462: {}'.format(x462))
        elif isinstance(x462, torch.Tensor):
            print('x462: {}'.format(x462.shape))
        elif isinstance(x462, tuple):
            tuple_shapes = '('
            for item in x462:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x462: {}'.format(tuple_shapes))
        else:
            print('x462: {}'.format(x462))
        x463=x462.contiguous()
        if x463 is None:
            print('x463: {}'.format(x463))
        elif isinstance(x463, torch.Tensor):
            print('x463: {}'.format(x463.shape))
        elif isinstance(x463, tuple):
            tuple_shapes = '('
            for item in x463:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x463: {}'.format(tuple_shapes))
        else:
            print('x463: {}'.format(x463))
        x464=x463.unsqueeze(0)
        if x464 is None:
            print('x464: {}'.format(x464))
        elif isinstance(x464, torch.Tensor):
            print('x464: {}'.format(x464.shape))
        elif isinstance(x464, tuple):
            tuple_shapes = '('
            for item in x464:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x464: {}'.format(tuple_shapes))
        else:
            print('x464: {}'.format(x464))
        x469=torchvision.models.swin_transformer.shifted_window_attention(x457, self.weight38, self.weight39, x464, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias38, proj_bias=self.bias39)
        if x469 is None:
            print('x469: {}'.format(x469))
        elif isinstance(x469, torch.Tensor):
            print('x469: {}'.format(x469.shape))
        elif isinstance(x469, tuple):
            tuple_shapes = '('
            for item in x469:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x469: {}'.format(tuple_shapes))
        else:
            print('x469: {}'.format(x469))
        x470=stochastic_depth(x469, 0.41304347826086957, 'row', False)
        if x470 is None:
            print('x470: {}'.format(x470))
        elif isinstance(x470, torch.Tensor):
            print('x470: {}'.format(x470.shape))
        elif isinstance(x470, tuple):
            tuple_shapes = '('
            for item in x470:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x470: {}'.format(tuple_shapes))
        else:
            print('x470: {}'.format(x470))
        x471=operator.add(x456, x470)
        if x471 is None:
            print('x471: {}'.format(x471))
        elif isinstance(x471, torch.Tensor):
            print('x471: {}'.format(x471.shape))
        elif isinstance(x471, tuple):
            tuple_shapes = '('
            for item in x471:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x471: {}'.format(tuple_shapes))
        else:
            print('x471: {}'.format(x471))
        x472=self.layernorm42(x471)
        if x472 is None:
            print('x472: {}'.format(x472))
        elif isinstance(x472, torch.Tensor):
            print('x472: {}'.format(x472.shape))
        elif isinstance(x472, tuple):
            tuple_shapes = '('
            for item in x472:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x472: {}'.format(tuple_shapes))
        else:
            print('x472: {}'.format(x472))
        x473=self.linear40(x472)
        if x473 is None:
            print('x473: {}'.format(x473))
        elif isinstance(x473, torch.Tensor):
            print('x473: {}'.format(x473.shape))
        elif isinstance(x473, tuple):
            tuple_shapes = '('
            for item in x473:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x473: {}'.format(tuple_shapes))
        else:
            print('x473: {}'.format(x473))
        x474=self.gelu19(x473)
        if x474 is None:
            print('x474: {}'.format(x474))
        elif isinstance(x474, torch.Tensor):
            print('x474: {}'.format(x474.shape))
        elif isinstance(x474, tuple):
            tuple_shapes = '('
            for item in x474:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x474: {}'.format(tuple_shapes))
        else:
            print('x474: {}'.format(x474))
        x475=self.dropout38(x474)
        if x475 is None:
            print('x475: {}'.format(x475))
        elif isinstance(x475, torch.Tensor):
            print('x475: {}'.format(x475.shape))
        elif isinstance(x475, tuple):
            tuple_shapes = '('
            for item in x475:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x475: {}'.format(tuple_shapes))
        else:
            print('x475: {}'.format(x475))
        x476=self.linear41(x475)
        if x476 is None:
            print('x476: {}'.format(x476))
        elif isinstance(x476, torch.Tensor):
            print('x476: {}'.format(x476.shape))
        elif isinstance(x476, tuple):
            tuple_shapes = '('
            for item in x476:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x476: {}'.format(tuple_shapes))
        else:
            print('x476: {}'.format(x476))
        x477=self.dropout39(x476)
        if x477 is None:
            print('x477: {}'.format(x477))
        elif isinstance(x477, torch.Tensor):
            print('x477: {}'.format(x477.shape))
        elif isinstance(x477, tuple):
            tuple_shapes = '('
            for item in x477:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x477: {}'.format(tuple_shapes))
        else:
            print('x477: {}'.format(x477))
        x478=stochastic_depth(x477, 0.41304347826086957, 'row', False)
        if x478 is None:
            print('x478: {}'.format(x478))
        elif isinstance(x478, torch.Tensor):
            print('x478: {}'.format(x478.shape))
        elif isinstance(x478, tuple):
            tuple_shapes = '('
            for item in x478:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x478: {}'.format(tuple_shapes))
        else:
            print('x478: {}'.format(x478))
        x479=operator.add(x471, x478)
        if x479 is None:
            print('x479: {}'.format(x479))
        elif isinstance(x479, torch.Tensor):
            print('x479: {}'.format(x479.shape))
        elif isinstance(x479, tuple):
            tuple_shapes = '('
            for item in x479:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x479: {}'.format(tuple_shapes))
        else:
            print('x479: {}'.format(x479))
        x480=self.layernorm43(x479)
        if x480 is None:
            print('x480: {}'.format(x480))
        elif isinstance(x480, torch.Tensor):
            print('x480: {}'.format(x480.shape))
        elif isinstance(x480, tuple):
            tuple_shapes = '('
            for item in x480:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x480: {}'.format(tuple_shapes))
        else:
            print('x480: {}'.format(x480))
        x483=operator.getitem(self.relative_position_bias_table20, self.relative_position_index20)
        if x483 is None:
            print('x483: {}'.format(x483))
        elif isinstance(x483, torch.Tensor):
            print('x483: {}'.format(x483.shape))
        elif isinstance(x483, tuple):
            tuple_shapes = '('
            for item in x483:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x483: {}'.format(tuple_shapes))
        else:
            print('x483: {}'.format(x483))
        x484=x483.view(49, 49, -1)
        if x484 is None:
            print('x484: {}'.format(x484))
        elif isinstance(x484, torch.Tensor):
            print('x484: {}'.format(x484.shape))
        elif isinstance(x484, tuple):
            tuple_shapes = '('
            for item in x484:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x484: {}'.format(tuple_shapes))
        else:
            print('x484: {}'.format(x484))
        x485=x484.permute(2, 0, 1)
        if x485 is None:
            print('x485: {}'.format(x485))
        elif isinstance(x485, torch.Tensor):
            print('x485: {}'.format(x485.shape))
        elif isinstance(x485, tuple):
            tuple_shapes = '('
            for item in x485:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x485: {}'.format(tuple_shapes))
        else:
            print('x485: {}'.format(x485))
        x486=x485.contiguous()
        if x486 is None:
            print('x486: {}'.format(x486))
        elif isinstance(x486, torch.Tensor):
            print('x486: {}'.format(x486.shape))
        elif isinstance(x486, tuple):
            tuple_shapes = '('
            for item in x486:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x486: {}'.format(tuple_shapes))
        else:
            print('x486: {}'.format(x486))
        x487=x486.unsqueeze(0)
        if x487 is None:
            print('x487: {}'.format(x487))
        elif isinstance(x487, torch.Tensor):
            print('x487: {}'.format(x487.shape))
        elif isinstance(x487, tuple):
            tuple_shapes = '('
            for item in x487:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x487: {}'.format(tuple_shapes))
        else:
            print('x487: {}'.format(x487))
        x492=torchvision.models.swin_transformer.shifted_window_attention(x480, self.weight40, self.weight41, x487, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias40, proj_bias=self.bias41)
        if x492 is None:
            print('x492: {}'.format(x492))
        elif isinstance(x492, torch.Tensor):
            print('x492: {}'.format(x492.shape))
        elif isinstance(x492, tuple):
            tuple_shapes = '('
            for item in x492:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x492: {}'.format(tuple_shapes))
        else:
            print('x492: {}'.format(x492))
        x493=stochastic_depth(x492, 0.43478260869565216, 'row', False)
        if x493 is None:
            print('x493: {}'.format(x493))
        elif isinstance(x493, torch.Tensor):
            print('x493: {}'.format(x493.shape))
        elif isinstance(x493, tuple):
            tuple_shapes = '('
            for item in x493:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x493: {}'.format(tuple_shapes))
        else:
            print('x493: {}'.format(x493))
        x494=operator.add(x479, x493)
        if x494 is None:
            print('x494: {}'.format(x494))
        elif isinstance(x494, torch.Tensor):
            print('x494: {}'.format(x494.shape))
        elif isinstance(x494, tuple):
            tuple_shapes = '('
            for item in x494:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x494: {}'.format(tuple_shapes))
        else:
            print('x494: {}'.format(x494))
        x495=self.layernorm44(x494)
        if x495 is None:
            print('x495: {}'.format(x495))
        elif isinstance(x495, torch.Tensor):
            print('x495: {}'.format(x495.shape))
        elif isinstance(x495, tuple):
            tuple_shapes = '('
            for item in x495:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x495: {}'.format(tuple_shapes))
        else:
            print('x495: {}'.format(x495))
        x496=self.linear42(x495)
        if x496 is None:
            print('x496: {}'.format(x496))
        elif isinstance(x496, torch.Tensor):
            print('x496: {}'.format(x496.shape))
        elif isinstance(x496, tuple):
            tuple_shapes = '('
            for item in x496:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x496: {}'.format(tuple_shapes))
        else:
            print('x496: {}'.format(x496))
        x497=self.gelu20(x496)
        if x497 is None:
            print('x497: {}'.format(x497))
        elif isinstance(x497, torch.Tensor):
            print('x497: {}'.format(x497.shape))
        elif isinstance(x497, tuple):
            tuple_shapes = '('
            for item in x497:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x497: {}'.format(tuple_shapes))
        else:
            print('x497: {}'.format(x497))
        x498=self.dropout40(x497)
        if x498 is None:
            print('x498: {}'.format(x498))
        elif isinstance(x498, torch.Tensor):
            print('x498: {}'.format(x498.shape))
        elif isinstance(x498, tuple):
            tuple_shapes = '('
            for item in x498:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x498: {}'.format(tuple_shapes))
        else:
            print('x498: {}'.format(x498))
        x499=self.linear43(x498)
        if x499 is None:
            print('x499: {}'.format(x499))
        elif isinstance(x499, torch.Tensor):
            print('x499: {}'.format(x499.shape))
        elif isinstance(x499, tuple):
            tuple_shapes = '('
            for item in x499:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x499: {}'.format(tuple_shapes))
        else:
            print('x499: {}'.format(x499))
        x500=self.dropout41(x499)
        if x500 is None:
            print('x500: {}'.format(x500))
        elif isinstance(x500, torch.Tensor):
            print('x500: {}'.format(x500.shape))
        elif isinstance(x500, tuple):
            tuple_shapes = '('
            for item in x500:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x500: {}'.format(tuple_shapes))
        else:
            print('x500: {}'.format(x500))
        x501=stochastic_depth(x500, 0.43478260869565216, 'row', False)
        if x501 is None:
            print('x501: {}'.format(x501))
        elif isinstance(x501, torch.Tensor):
            print('x501: {}'.format(x501.shape))
        elif isinstance(x501, tuple):
            tuple_shapes = '('
            for item in x501:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x501: {}'.format(tuple_shapes))
        else:
            print('x501: {}'.format(x501))
        x502=operator.add(x494, x501)
        if x502 is None:
            print('x502: {}'.format(x502))
        elif isinstance(x502, torch.Tensor):
            print('x502: {}'.format(x502.shape))
        elif isinstance(x502, tuple):
            tuple_shapes = '('
            for item in x502:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x502: {}'.format(tuple_shapes))
        else:
            print('x502: {}'.format(x502))
        x503=self.layernorm45(x502)
        if x503 is None:
            print('x503: {}'.format(x503))
        elif isinstance(x503, torch.Tensor):
            print('x503: {}'.format(x503.shape))
        elif isinstance(x503, tuple):
            tuple_shapes = '('
            for item in x503:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x503: {}'.format(tuple_shapes))
        else:
            print('x503: {}'.format(x503))
        x506=operator.getitem(self.relative_position_bias_table21, self.relative_position_index21)
        if x506 is None:
            print('x506: {}'.format(x506))
        elif isinstance(x506, torch.Tensor):
            print('x506: {}'.format(x506.shape))
        elif isinstance(x506, tuple):
            tuple_shapes = '('
            for item in x506:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x506: {}'.format(tuple_shapes))
        else:
            print('x506: {}'.format(x506))
        x507=x506.view(49, 49, -1)
        if x507 is None:
            print('x507: {}'.format(x507))
        elif isinstance(x507, torch.Tensor):
            print('x507: {}'.format(x507.shape))
        elif isinstance(x507, tuple):
            tuple_shapes = '('
            for item in x507:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x507: {}'.format(tuple_shapes))
        else:
            print('x507: {}'.format(x507))
        x508=x507.permute(2, 0, 1)
        if x508 is None:
            print('x508: {}'.format(x508))
        elif isinstance(x508, torch.Tensor):
            print('x508: {}'.format(x508.shape))
        elif isinstance(x508, tuple):
            tuple_shapes = '('
            for item in x508:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x508: {}'.format(tuple_shapes))
        else:
            print('x508: {}'.format(x508))
        x509=x508.contiguous()
        if x509 is None:
            print('x509: {}'.format(x509))
        elif isinstance(x509, torch.Tensor):
            print('x509: {}'.format(x509.shape))
        elif isinstance(x509, tuple):
            tuple_shapes = '('
            for item in x509:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x509: {}'.format(tuple_shapes))
        else:
            print('x509: {}'.format(x509))
        x510=x509.unsqueeze(0)
        if x510 is None:
            print('x510: {}'.format(x510))
        elif isinstance(x510, torch.Tensor):
            print('x510: {}'.format(x510.shape))
        elif isinstance(x510, tuple):
            tuple_shapes = '('
            for item in x510:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x510: {}'.format(tuple_shapes))
        else:
            print('x510: {}'.format(x510))
        x515=torchvision.models.swin_transformer.shifted_window_attention(x503, self.weight42, self.weight43, x510, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias42, proj_bias=self.bias43)
        if x515 is None:
            print('x515: {}'.format(x515))
        elif isinstance(x515, torch.Tensor):
            print('x515: {}'.format(x515.shape))
        elif isinstance(x515, tuple):
            tuple_shapes = '('
            for item in x515:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x515: {}'.format(tuple_shapes))
        else:
            print('x515: {}'.format(x515))
        x516=stochastic_depth(x515, 0.45652173913043476, 'row', False)
        if x516 is None:
            print('x516: {}'.format(x516))
        elif isinstance(x516, torch.Tensor):
            print('x516: {}'.format(x516.shape))
        elif isinstance(x516, tuple):
            tuple_shapes = '('
            for item in x516:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x516: {}'.format(tuple_shapes))
        else:
            print('x516: {}'.format(x516))
        x517=operator.add(x502, x516)
        if x517 is None:
            print('x517: {}'.format(x517))
        elif isinstance(x517, torch.Tensor):
            print('x517: {}'.format(x517.shape))
        elif isinstance(x517, tuple):
            tuple_shapes = '('
            for item in x517:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x517: {}'.format(tuple_shapes))
        else:
            print('x517: {}'.format(x517))
        x518=self.layernorm46(x517)
        if x518 is None:
            print('x518: {}'.format(x518))
        elif isinstance(x518, torch.Tensor):
            print('x518: {}'.format(x518.shape))
        elif isinstance(x518, tuple):
            tuple_shapes = '('
            for item in x518:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x518: {}'.format(tuple_shapes))
        else:
            print('x518: {}'.format(x518))
        x519=self.linear44(x518)
        if x519 is None:
            print('x519: {}'.format(x519))
        elif isinstance(x519, torch.Tensor):
            print('x519: {}'.format(x519.shape))
        elif isinstance(x519, tuple):
            tuple_shapes = '('
            for item in x519:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x519: {}'.format(tuple_shapes))
        else:
            print('x519: {}'.format(x519))
        x520=self.gelu21(x519)
        if x520 is None:
            print('x520: {}'.format(x520))
        elif isinstance(x520, torch.Tensor):
            print('x520: {}'.format(x520.shape))
        elif isinstance(x520, tuple):
            tuple_shapes = '('
            for item in x520:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x520: {}'.format(tuple_shapes))
        else:
            print('x520: {}'.format(x520))
        x521=self.dropout42(x520)
        if x521 is None:
            print('x521: {}'.format(x521))
        elif isinstance(x521, torch.Tensor):
            print('x521: {}'.format(x521.shape))
        elif isinstance(x521, tuple):
            tuple_shapes = '('
            for item in x521:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x521: {}'.format(tuple_shapes))
        else:
            print('x521: {}'.format(x521))
        x522=self.linear45(x521)
        if x522 is None:
            print('x522: {}'.format(x522))
        elif isinstance(x522, torch.Tensor):
            print('x522: {}'.format(x522.shape))
        elif isinstance(x522, tuple):
            tuple_shapes = '('
            for item in x522:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x522: {}'.format(tuple_shapes))
        else:
            print('x522: {}'.format(x522))
        x523=self.dropout43(x522)
        if x523 is None:
            print('x523: {}'.format(x523))
        elif isinstance(x523, torch.Tensor):
            print('x523: {}'.format(x523.shape))
        elif isinstance(x523, tuple):
            tuple_shapes = '('
            for item in x523:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x523: {}'.format(tuple_shapes))
        else:
            print('x523: {}'.format(x523))
        x524=stochastic_depth(x523, 0.45652173913043476, 'row', False)
        if x524 is None:
            print('x524: {}'.format(x524))
        elif isinstance(x524, torch.Tensor):
            print('x524: {}'.format(x524.shape))
        elif isinstance(x524, tuple):
            tuple_shapes = '('
            for item in x524:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x524: {}'.format(tuple_shapes))
        else:
            print('x524: {}'.format(x524))
        x525=operator.add(x517, x524)
        if x525 is None:
            print('x525: {}'.format(x525))
        elif isinstance(x525, torch.Tensor):
            print('x525: {}'.format(x525.shape))
        elif isinstance(x525, tuple):
            tuple_shapes = '('
            for item in x525:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x525: {}'.format(tuple_shapes))
        else:
            print('x525: {}'.format(x525))
        x526=torchvision.models.swin_transformer._patch_merging_pad(x525)
        if x526 is None:
            print('x526: {}'.format(x526))
        elif isinstance(x526, torch.Tensor):
            print('x526: {}'.format(x526.shape))
        elif isinstance(x526, tuple):
            tuple_shapes = '('
            for item in x526:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x526: {}'.format(tuple_shapes))
        else:
            print('x526: {}'.format(x526))
        x527=operator.getitem(x526, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x527 is None:
            print('x527: {}'.format(x527))
        elif isinstance(x527, torch.Tensor):
            print('x527: {}'.format(x527.shape))
        elif isinstance(x527, tuple):
            tuple_shapes = '('
            for item in x527:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x527: {}'.format(tuple_shapes))
        else:
            print('x527: {}'.format(x527))
        x528=operator.getitem(x526, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x528 is None:
            print('x528: {}'.format(x528))
        elif isinstance(x528, torch.Tensor):
            print('x528: {}'.format(x528.shape))
        elif isinstance(x528, tuple):
            tuple_shapes = '('
            for item in x528:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x528: {}'.format(tuple_shapes))
        else:
            print('x528: {}'.format(x528))
        x529=operator.getitem(x526, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x529 is None:
            print('x529: {}'.format(x529))
        elif isinstance(x529, torch.Tensor):
            print('x529: {}'.format(x529.shape))
        elif isinstance(x529, tuple):
            tuple_shapes = '('
            for item in x529:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x529: {}'.format(tuple_shapes))
        else:
            print('x529: {}'.format(x529))
        x530=operator.getitem(x526, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x530 is None:
            print('x530: {}'.format(x530))
        elif isinstance(x530, torch.Tensor):
            print('x530: {}'.format(x530.shape))
        elif isinstance(x530, tuple):
            tuple_shapes = '('
            for item in x530:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x530: {}'.format(tuple_shapes))
        else:
            print('x530: {}'.format(x530))
        x531=torch.cat([x527, x528, x529, x530], -1)
        if x531 is None:
            print('x531: {}'.format(x531))
        elif isinstance(x531, torch.Tensor):
            print('x531: {}'.format(x531.shape))
        elif isinstance(x531, tuple):
            tuple_shapes = '('
            for item in x531:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x531: {}'.format(tuple_shapes))
        else:
            print('x531: {}'.format(x531))
        x532=self.layernorm47(x531)
        if x532 is None:
            print('x532: {}'.format(x532))
        elif isinstance(x532, torch.Tensor):
            print('x532: {}'.format(x532.shape))
        elif isinstance(x532, tuple):
            tuple_shapes = '('
            for item in x532:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x532: {}'.format(tuple_shapes))
        else:
            print('x532: {}'.format(x532))
        x533=self.linear46(x532)
        if x533 is None:
            print('x533: {}'.format(x533))
        elif isinstance(x533, torch.Tensor):
            print('x533: {}'.format(x533.shape))
        elif isinstance(x533, tuple):
            tuple_shapes = '('
            for item in x533:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x533: {}'.format(tuple_shapes))
        else:
            print('x533: {}'.format(x533))
        x534=self.layernorm48(x533)
        if x534 is None:
            print('x534: {}'.format(x534))
        elif isinstance(x534, torch.Tensor):
            print('x534: {}'.format(x534.shape))
        elif isinstance(x534, tuple):
            tuple_shapes = '('
            for item in x534:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x534: {}'.format(tuple_shapes))
        else:
            print('x534: {}'.format(x534))
        x537=operator.getitem(self.relative_position_bias_table22, self.relative_position_index22)
        if x537 is None:
            print('x537: {}'.format(x537))
        elif isinstance(x537, torch.Tensor):
            print('x537: {}'.format(x537.shape))
        elif isinstance(x537, tuple):
            tuple_shapes = '('
            for item in x537:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x537: {}'.format(tuple_shapes))
        else:
            print('x537: {}'.format(x537))
        x538=x537.view(49, 49, -1)
        if x538 is None:
            print('x538: {}'.format(x538))
        elif isinstance(x538, torch.Tensor):
            print('x538: {}'.format(x538.shape))
        elif isinstance(x538, tuple):
            tuple_shapes = '('
            for item in x538:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x538: {}'.format(tuple_shapes))
        else:
            print('x538: {}'.format(x538))
        x539=x538.permute(2, 0, 1)
        if x539 is None:
            print('x539: {}'.format(x539))
        elif isinstance(x539, torch.Tensor):
            print('x539: {}'.format(x539.shape))
        elif isinstance(x539, tuple):
            tuple_shapes = '('
            for item in x539:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x539: {}'.format(tuple_shapes))
        else:
            print('x539: {}'.format(x539))
        x540=x539.contiguous()
        if x540 is None:
            print('x540: {}'.format(x540))
        elif isinstance(x540, torch.Tensor):
            print('x540: {}'.format(x540.shape))
        elif isinstance(x540, tuple):
            tuple_shapes = '('
            for item in x540:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x540: {}'.format(tuple_shapes))
        else:
            print('x540: {}'.format(x540))
        x541=x540.unsqueeze(0)
        if x541 is None:
            print('x541: {}'.format(x541))
        elif isinstance(x541, torch.Tensor):
            print('x541: {}'.format(x541.shape))
        elif isinstance(x541, tuple):
            tuple_shapes = '('
            for item in x541:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x541: {}'.format(tuple_shapes))
        else:
            print('x541: {}'.format(x541))
        x546=torchvision.models.swin_transformer.shifted_window_attention(x534, self.weight44, self.weight45, x541, [7, 7], 32,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias44, proj_bias=self.bias45)
        if x546 is None:
            print('x546: {}'.format(x546))
        elif isinstance(x546, torch.Tensor):
            print('x546: {}'.format(x546.shape))
        elif isinstance(x546, tuple):
            tuple_shapes = '('
            for item in x546:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x546: {}'.format(tuple_shapes))
        else:
            print('x546: {}'.format(x546))
        x547=stochastic_depth(x546, 0.4782608695652174, 'row', False)
        if x547 is None:
            print('x547: {}'.format(x547))
        elif isinstance(x547, torch.Tensor):
            print('x547: {}'.format(x547.shape))
        elif isinstance(x547, tuple):
            tuple_shapes = '('
            for item in x547:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x547: {}'.format(tuple_shapes))
        else:
            print('x547: {}'.format(x547))
        x548=operator.add(x533, x547)
        if x548 is None:
            print('x548: {}'.format(x548))
        elif isinstance(x548, torch.Tensor):
            print('x548: {}'.format(x548.shape))
        elif isinstance(x548, tuple):
            tuple_shapes = '('
            for item in x548:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x548: {}'.format(tuple_shapes))
        else:
            print('x548: {}'.format(x548))
        x549=self.layernorm49(x548)
        if x549 is None:
            print('x549: {}'.format(x549))
        elif isinstance(x549, torch.Tensor):
            print('x549: {}'.format(x549.shape))
        elif isinstance(x549, tuple):
            tuple_shapes = '('
            for item in x549:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x549: {}'.format(tuple_shapes))
        else:
            print('x549: {}'.format(x549))
        x550=self.linear47(x549)
        if x550 is None:
            print('x550: {}'.format(x550))
        elif isinstance(x550, torch.Tensor):
            print('x550: {}'.format(x550.shape))
        elif isinstance(x550, tuple):
            tuple_shapes = '('
            for item in x550:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x550: {}'.format(tuple_shapes))
        else:
            print('x550: {}'.format(x550))
        x551=self.gelu22(x550)
        if x551 is None:
            print('x551: {}'.format(x551))
        elif isinstance(x551, torch.Tensor):
            print('x551: {}'.format(x551.shape))
        elif isinstance(x551, tuple):
            tuple_shapes = '('
            for item in x551:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x551: {}'.format(tuple_shapes))
        else:
            print('x551: {}'.format(x551))
        x552=self.dropout44(x551)
        if x552 is None:
            print('x552: {}'.format(x552))
        elif isinstance(x552, torch.Tensor):
            print('x552: {}'.format(x552.shape))
        elif isinstance(x552, tuple):
            tuple_shapes = '('
            for item in x552:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x552: {}'.format(tuple_shapes))
        else:
            print('x552: {}'.format(x552))
        x553=self.linear48(x552)
        if x553 is None:
            print('x553: {}'.format(x553))
        elif isinstance(x553, torch.Tensor):
            print('x553: {}'.format(x553.shape))
        elif isinstance(x553, tuple):
            tuple_shapes = '('
            for item in x553:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x553: {}'.format(tuple_shapes))
        else:
            print('x553: {}'.format(x553))
        x554=self.dropout45(x553)
        if x554 is None:
            print('x554: {}'.format(x554))
        elif isinstance(x554, torch.Tensor):
            print('x554: {}'.format(x554.shape))
        elif isinstance(x554, tuple):
            tuple_shapes = '('
            for item in x554:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x554: {}'.format(tuple_shapes))
        else:
            print('x554: {}'.format(x554))
        x555=stochastic_depth(x554, 0.4782608695652174, 'row', False)
        if x555 is None:
            print('x555: {}'.format(x555))
        elif isinstance(x555, torch.Tensor):
            print('x555: {}'.format(x555.shape))
        elif isinstance(x555, tuple):
            tuple_shapes = '('
            for item in x555:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x555: {}'.format(tuple_shapes))
        else:
            print('x555: {}'.format(x555))
        x556=operator.add(x548, x555)
        if x556 is None:
            print('x556: {}'.format(x556))
        elif isinstance(x556, torch.Tensor):
            print('x556: {}'.format(x556.shape))
        elif isinstance(x556, tuple):
            tuple_shapes = '('
            for item in x556:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x556: {}'.format(tuple_shapes))
        else:
            print('x556: {}'.format(x556))
        x557=self.layernorm50(x556)
        if x557 is None:
            print('x557: {}'.format(x557))
        elif isinstance(x557, torch.Tensor):
            print('x557: {}'.format(x557.shape))
        elif isinstance(x557, tuple):
            tuple_shapes = '('
            for item in x557:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x557: {}'.format(tuple_shapes))
        else:
            print('x557: {}'.format(x557))
        x560=operator.getitem(self.relative_position_bias_table23, self.relative_position_index23)
        if x560 is None:
            print('x560: {}'.format(x560))
        elif isinstance(x560, torch.Tensor):
            print('x560: {}'.format(x560.shape))
        elif isinstance(x560, tuple):
            tuple_shapes = '('
            for item in x560:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x560: {}'.format(tuple_shapes))
        else:
            print('x560: {}'.format(x560))
        x561=x560.view(49, 49, -1)
        if x561 is None:
            print('x561: {}'.format(x561))
        elif isinstance(x561, torch.Tensor):
            print('x561: {}'.format(x561.shape))
        elif isinstance(x561, tuple):
            tuple_shapes = '('
            for item in x561:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x561: {}'.format(tuple_shapes))
        else:
            print('x561: {}'.format(x561))
        x562=x561.permute(2, 0, 1)
        if x562 is None:
            print('x562: {}'.format(x562))
        elif isinstance(x562, torch.Tensor):
            print('x562: {}'.format(x562.shape))
        elif isinstance(x562, tuple):
            tuple_shapes = '('
            for item in x562:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x562: {}'.format(tuple_shapes))
        else:
            print('x562: {}'.format(x562))
        x563=x562.contiguous()
        if x563 is None:
            print('x563: {}'.format(x563))
        elif isinstance(x563, torch.Tensor):
            print('x563: {}'.format(x563.shape))
        elif isinstance(x563, tuple):
            tuple_shapes = '('
            for item in x563:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x563: {}'.format(tuple_shapes))
        else:
            print('x563: {}'.format(x563))
        x564=x563.unsqueeze(0)
        if x564 is None:
            print('x564: {}'.format(x564))
        elif isinstance(x564, torch.Tensor):
            print('x564: {}'.format(x564.shape))
        elif isinstance(x564, tuple):
            tuple_shapes = '('
            for item in x564:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x564: {}'.format(tuple_shapes))
        else:
            print('x564: {}'.format(x564))
        x569=torchvision.models.swin_transformer.shifted_window_attention(x557, self.weight46, self.weight47, x564, [7, 7], 32,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias46, proj_bias=self.bias47)
        if x569 is None:
            print('x569: {}'.format(x569))
        elif isinstance(x569, torch.Tensor):
            print('x569: {}'.format(x569.shape))
        elif isinstance(x569, tuple):
            tuple_shapes = '('
            for item in x569:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x569: {}'.format(tuple_shapes))
        else:
            print('x569: {}'.format(x569))
        x570=stochastic_depth(x569, 0.5, 'row', False)
        if x570 is None:
            print('x570: {}'.format(x570))
        elif isinstance(x570, torch.Tensor):
            print('x570: {}'.format(x570.shape))
        elif isinstance(x570, tuple):
            tuple_shapes = '('
            for item in x570:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x570: {}'.format(tuple_shapes))
        else:
            print('x570: {}'.format(x570))
        x571=operator.add(x556, x570)
        if x571 is None:
            print('x571: {}'.format(x571))
        elif isinstance(x571, torch.Tensor):
            print('x571: {}'.format(x571.shape))
        elif isinstance(x571, tuple):
            tuple_shapes = '('
            for item in x571:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x571: {}'.format(tuple_shapes))
        else:
            print('x571: {}'.format(x571))
        x572=self.layernorm51(x571)
        if x572 is None:
            print('x572: {}'.format(x572))
        elif isinstance(x572, torch.Tensor):
            print('x572: {}'.format(x572.shape))
        elif isinstance(x572, tuple):
            tuple_shapes = '('
            for item in x572:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x572: {}'.format(tuple_shapes))
        else:
            print('x572: {}'.format(x572))
        x573=self.linear49(x572)
        if x573 is None:
            print('x573: {}'.format(x573))
        elif isinstance(x573, torch.Tensor):
            print('x573: {}'.format(x573.shape))
        elif isinstance(x573, tuple):
            tuple_shapes = '('
            for item in x573:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x573: {}'.format(tuple_shapes))
        else:
            print('x573: {}'.format(x573))
        x574=self.gelu23(x573)
        if x574 is None:
            print('x574: {}'.format(x574))
        elif isinstance(x574, torch.Tensor):
            print('x574: {}'.format(x574.shape))
        elif isinstance(x574, tuple):
            tuple_shapes = '('
            for item in x574:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x574: {}'.format(tuple_shapes))
        else:
            print('x574: {}'.format(x574))
        x575=self.dropout46(x574)
        if x575 is None:
            print('x575: {}'.format(x575))
        elif isinstance(x575, torch.Tensor):
            print('x575: {}'.format(x575.shape))
        elif isinstance(x575, tuple):
            tuple_shapes = '('
            for item in x575:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x575: {}'.format(tuple_shapes))
        else:
            print('x575: {}'.format(x575))
        x576=self.linear50(x575)
        if x576 is None:
            print('x576: {}'.format(x576))
        elif isinstance(x576, torch.Tensor):
            print('x576: {}'.format(x576.shape))
        elif isinstance(x576, tuple):
            tuple_shapes = '('
            for item in x576:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x576: {}'.format(tuple_shapes))
        else:
            print('x576: {}'.format(x576))
        x577=self.dropout47(x576)
        if x577 is None:
            print('x577: {}'.format(x577))
        elif isinstance(x577, torch.Tensor):
            print('x577: {}'.format(x577.shape))
        elif isinstance(x577, tuple):
            tuple_shapes = '('
            for item in x577:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x577: {}'.format(tuple_shapes))
        else:
            print('x577: {}'.format(x577))
        x578=stochastic_depth(x577, 0.5, 'row', False)
        if x578 is None:
            print('x578: {}'.format(x578))
        elif isinstance(x578, torch.Tensor):
            print('x578: {}'.format(x578.shape))
        elif isinstance(x578, tuple):
            tuple_shapes = '('
            for item in x578:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x578: {}'.format(tuple_shapes))
        else:
            print('x578: {}'.format(x578))
        x579=operator.add(x571, x578)
        if x579 is None:
            print('x579: {}'.format(x579))
        elif isinstance(x579, torch.Tensor):
            print('x579: {}'.format(x579.shape))
        elif isinstance(x579, tuple):
            tuple_shapes = '('
            for item in x579:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x579: {}'.format(tuple_shapes))
        else:
            print('x579: {}'.format(x579))
        x580=self.layernorm52(x579)
        if x580 is None:
            print('x580: {}'.format(x580))
        elif isinstance(x580, torch.Tensor):
            print('x580: {}'.format(x580.shape))
        elif isinstance(x580, tuple):
            tuple_shapes = '('
            for item in x580:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x580: {}'.format(tuple_shapes))
        else:
            print('x580: {}'.format(x580))
        x581=x580.permute(0, 3, 1, 2)
        if x581 is None:
            print('x581: {}'.format(x581))
        elif isinstance(x581, torch.Tensor):
            print('x581: {}'.format(x581.shape))
        elif isinstance(x581, tuple):
            tuple_shapes = '('
            for item in x581:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x581: {}'.format(tuple_shapes))
        else:
            print('x581: {}'.format(x581))
        x582=self.adaptiveavgpool2d0(x581)
        if x582 is None:
            print('x582: {}'.format(x582))
        elif isinstance(x582, torch.Tensor):
            print('x582: {}'.format(x582.shape))
        elif isinstance(x582, tuple):
            tuple_shapes = '('
            for item in x582:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x582: {}'.format(tuple_shapes))
        else:
            print('x582: {}'.format(x582))
        x583=torch.flatten(x582, 1)
        if x583 is None:
            print('x583: {}'.format(x583))
        elif isinstance(x583, torch.Tensor):
            print('x583: {}'.format(x583.shape))
        elif isinstance(x583, tuple):
            tuple_shapes = '('
            for item in x583:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x583: {}'.format(tuple_shapes))
        else:
            print('x583: {}'.format(x583))
        x584=self.linear51(x583)
        if x584 is None:
            print('x584: {}'.format(x584))
        elif isinstance(x584, torch.Tensor):
            print('x584: {}'.format(x584.shape))
        elif isinstance(x584, tuple):
            tuple_shapes = '('
            for item in x584:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x584: {}'.format(tuple_shapes))
        else:
            print('x584: {}'.format(x584))

m = M().eval()
x = torch.rand(1, 3, 224, 224)
output = m(x)
