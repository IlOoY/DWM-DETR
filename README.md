# DWM-DETR — 三个核心模块

本仓库只提供三个改进模块及其必要的局部辅助定义，不包含完整检测器、训练框架、数据集、权重或实验日志。

| 模块 | 核心代码 | 内容 |
| --- | --- | --- |
| 末级可变形残差模块 | [dcn_stage5.py](dcn_stage5.py) | DCNv2、残差连接以及R18末级两个残差块，共四个3×3可变形卷积 |
| 双流小波下采样 DSWD | [dswd.py](dswd.py) | 空间卷积分支、完整Haar四子带分支及仅对小波输出施加的可学习标量 |
| 共享尺度专家编码 MoSME | [mosme.py](mosme.py) | 常开共享专家、逐令牌Top-2路由、层级嵌入、共享注意力和出口归一化 |

代码从提供的本地实现提取，保留核心类的计算逻辑，补入必要的激活函数辅助定义。两个构造函数用于显示原模型的模块组装方式。

## 依赖

需要相互兼容的 `torch` 和 `torchvision`，且torchvision包含可用的 `deform_conv2d` 算子。主专家实现中的 `torch_scatter` 为可选依赖，保留原实现的PyTorch聚合回退路径。

## 使用方式

```python
from dcn_stage5 import build_deformable_stage5
from dswd import DualStreamDownsample
from mosme import build_shared_scale_encoder

stage5 = build_deformable_stage5(256, 512)
downsample = DualStreamDownsample(256, 256)
encoder = build_shared_scale_encoder()

# tokens_s4和tokens_s5分别为[B, N4, 256]与[B, N5, 256]。
# 两个尺度分别调用同一个encoder实例，实现参数共享。
out_s4, loss_s4 = encoder(tokens_s4, scale_index=0, pos_embed=pos_s4)
out_s5, loss_s5 = encoder(tokens_s5, scale_index=1, pos_embed=pos_s5)
```

DSWD输入的高、宽应为偶数。MoSME返回的均衡项需由调用方按实际训练方案汇总；此处不提供训练循环。

已核对Python语法、核心定义与原源码的AST一致性及局部依赖。本次未运行GPU训练或算子运行测试。

## 来源

基础实现来自 [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)，保留相关版权说明。上游许可证条款及本地修改说明见[NOTICE.md](NOTICE.md)。本次未为项目改动新增授权许可。
