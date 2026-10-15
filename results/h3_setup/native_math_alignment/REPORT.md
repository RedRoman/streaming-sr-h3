# 原生训练与官方采样数学审计（v1）

固定训练源 bear_256、22帧、同真实中性text缓存，normal50 sigma网格index46：σv=0.43902439、σa=0.16363637、t=439.02438。仅用HR latent构造监督噪声桥，用于数学审计；此输出不是可因果使用的SR读出。

实测v目标为epsilon−zHR，CONST去噪为xσ−σv；解析perfect-v回到HR maxabs2.38e−7，反号对照RMSE1.2844。video process_in/out maxabs0、scale1；24通道guide、布局分段和text预处理完全相同。零adapter与guide末层单bias+0.01两个条件，对齐实际wrapper的BF16 video、BF16 carried audio再转换physical audio后，raw与官方velocity预测均maxabs0；一阶Euler相对xσ−σv maxabs2.38e−7。单bias扰动影响官方x0，RMSE0.01136，证明确实到达条件路径。

发现并保留的v1混杂：raw训练流输入/输出为FP32，官方BaseModel包装层为BF16；零adapter velocity RMSE0.025746、最大差0.45219，对应x0 RMSE0.011303、最大差0.19852。输入video rounding RMSE0.001219，audio physical rounding RMSE0.000414。不能称训练和采样完全数值等价，也不能只把raw漂移归因训练集小。

训练condition seed703、采样seed101导致条件行噪声RMSE0.001412；这是既有固定不同噪声协议，审计比较时统一703。其guide前处理CPU/GPU本身maxabs0。没有发现time/符号/scale/guide路由的错误；精度差异应在训练入口修复，而不是继续旧权重增加步数。

前台session40557实际terminal exit0；运行16.81s、峰值allocated40,690,521,088 bytes，冻结基础参数版本不变、无基础参数梯度。源码SHA、输入SHA、完整分条件数值见result.json；诊断latents和log均已本地归档。GPU6最后实测free81110MiB/util0，无该PID。

Root已授权独立native_io_v2，用与官方相同的BF16流IO、FP32 loss，先审计数学与真反传，再从同初始化/seed/case/sigma/noise/lr重新各400；旧native_sr_pilot400保留为legacy_fp32_io，不混并为v2结果。
