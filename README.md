[TOC]
# 1. 绪言
&emsp;&emsp;原始的代码要在 opencv2.4 上才能跑，更改至opencv4, 将特征提取改一下，也可以兼容opencv3。
之前在做AVM与透明底盘的时候，拼接算法中的一些思想也有用到，正好借[VideoStitch](https://github.com/MengLiPKU/VideoStitch),
把里面涉及的算法整理一下，当然，代码依然很乱。拼接流程如下图：
![stitching_pipeline](./doc/StitchingPipeline.jpg)

# 2. 离线过程
&emsp;&emsp;离线过程对应 Registration 过程，主要是输入首帧图像，得到融合区域、融合权重、图像之间的拼接顺序等信息，然后存储下来，
待实时拼接的时候就可以直接进行像素的映射与加权融合。离线过程也是计算量最大的，通常不会放到实时拼接过程中。

# 3. 实时拼接
&emsp;&emsp;实时拼接过程中对应 Compositing 过程，在离线标定获取到拼接信息以后，一般直接通过查表进行像素赋值是比较慢的，针对平台用opengl、neon
进行加速可以显著提升拼接速度。此过程还需进行光照补偿，保持画面的一致性，这里暂时没加，
所以拼接结果还是有点问题的，有待完善。

# 4. 测试素材
&emsp;&emsp;对视视频的同步性要求比较高，暂时使用一个视频裁剪来用。

# 5. build
## 5.1 ubuntu
&emsp;&emsp;安装opencv4以后，可以在ubuntu下直接通过cmake编译运行

## 5.2 windows
&emsp;&emsp;根据opencv.cmake设置opencv路径以及opencv版本，再使用cmake编译运行
```
set(OpenCV_INCLUDE_DIRS ${THIRD_PARTY_PATH}/opencv/include)
set(Opencv_LIB_DIRS ${THIRD_PARTY_PATH}/opencv/lib)
```