[TOC]
# 1. ����
&emsp;&emsp;ԭʼ�Ĵ���Ҫ�� opencv2.4 �ϲ����ܣ�������opencv4, ��������ȡ��һ�£�Ҳ���Լ���opencv3��
֮ǰ����AVM��͸�����̵�ʱ��ƴ���㷨�е�һЩ˼��Ҳ���õ������ý�[VideoStitch](https://github.com/MengLiPKU/VideoStitch),
�������漰���㷨����һ�£���Ȼ��������Ȼ���ҡ�ƴ����������ͼ��
![stitching_pipeline](./doc/StitchingPipeline.jpg)

# 2. ���߹���
&emsp;&emsp;���߹��̶�Ӧ Registration ���̣���Ҫ��������֡ͼ�񣬵õ��ں������ں�Ȩ�ء�ͼ��֮���ƴ��˳�����Ϣ��Ȼ��洢������
��ʵʱƴ�ӵ�ʱ��Ϳ���ֱ�ӽ������ص�ӳ�����Ȩ�ںϡ����߹���Ҳ�Ǽ��������ģ�ͨ������ŵ�ʵʱƴ�ӹ����С�

# 3. ʵʱƴ��
&emsp;&emsp;ʵʱƴ�ӹ����ж�Ӧ Compositing ���̣������߱궨��ȡ��ƴ����Ϣ�Ժ�һ��ֱ��ͨ�����������ظ�ֵ�ǱȽ����ģ����ƽ̨��opengl��neon
���м��ٿ�����������ƴ���ٶȡ��˹��̻�����й��ղ��������ֻ����һ���ԣ�������ʱû�ӣ�
����ƴ�ӽ�������е�����ģ��д����ơ�

# 4. �����ز�
&emsp;&emsp;������Ƶ��ͬ����Ҫ��Ƚϸߣ���ʱʹ��һ����Ƶ�ü����á�

# 5. build
## 5.1 ubuntu
&emsp;&emsp;��װopencv4�Ժ󣬿�����ubuntu��ֱ��ͨ��cmake��������

## 5.2 windows
&emsp;&emsp;����opencv.cmake����opencv·���Լ�opencv�汾����ʹ��cmake��������
```
set(OpenCV_INCLUDE_DIRS ${THIRD_PARTY_PATH}/opencv/include)
set(Opencv_LIB_DIRS ${THIRD_PARTY_PATH}/opencv/lib)
```