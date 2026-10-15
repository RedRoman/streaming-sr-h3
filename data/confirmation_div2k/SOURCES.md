# 新来源确认集：DIV2K 32张，方法冻结前暂不评价

来源为[DIV2K官方数据页](https://data.vision.ee.ethz.ch/cvl/DIV2K/)及其直接提供的[validation HR archive](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)。官方页面说明该集合有100张高分辨率验证图，供学术研究使用，图像版权属于原所有者；使用时引用Agustsson与Timofte的NTIRE 2017数据论文。这里下载用于本项目研究评价，不改称官方隐藏test。

为何另备数据：早期`results/learned_core/main/manifest.json`已经包含BSDS全部200 train / 100 val / 200 test，且test用于早期小模型评价。因此，即使某张BSDS图片未进入后续H3缓存，也不能称本项目从未使用的确认来源。

预先固定选择：按`SHA256('h3-confirmation-20261015-v1/' + basename)`对100个官方validation文件名排序，取前32个。不读取像素内容、不按质量、纹理、指标或模型输出选样。准确名单与archive ETag/字节数/central-directory hash在`sealed_plan.json`与`archive_identity.json`。

准备阶段只检查ZIP CRC、原始文件SHA、PNG签名与IHDR尺寸，不解码、观看、裁剪、训练或评估。所有文件保留在`sealed_HR/`，成功与否看`manifest.json`及`LIVE_STATE.json`，不能把下载中的状态算已完成。方法、checkpoint、退化与评价协议固定后，才启动这组确认。

这是项目新数据集来源，不保证H3预训练没有见过。暂未核验跨数据集的感知近重复，完整官方archive也没有全部下载；采用同一ETag的HTTP范围读取与逐成员CRC/SHA验证。源图像不作为真实运动视频；后续若用于合成camera episode，应明确仿真来源。真实长视频确认仍须另做。

复现脚本：`experiments/real_video_data/prepare_confirmation.py`。

本轮32张原始PNG共137,941,164字节，均只保存在本机sealed_HR；尚未复制图像到H100。若仅同步了元数据，不代表远端已有对应图像。
