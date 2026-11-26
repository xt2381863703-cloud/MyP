# 旋转毫米波雷达统一重建

唯一成像入口为：

```bash
python -m code.image_scene
```

一次重建输出同一套最终表面、逐 chirp 轨迹和 RF 回波法线，默认生成三幅中文
对比图。原法线、水平轮廓的独立入口及配置已移除，运行时不再重复执行不同的表面
选择流程。`paper/`、`pano/` 不参与运行或源码指纹；无神经网络训练或权重加载。

## 数据和运行配置

[config/data.yaml](config/data.yaml) 统一设置数据类型与单条输入轨迹：

```yaml
dataset_type: SelfData
in_traj_folder: /mnt/dataset/suradar/raw/8#402_moving/exp260521_000
```

切换原 PanoRadar 数据时，同时修改为：

```yaml
dataset_type: PanoData
in_traj_folder: /mnt/dataset/panoradar/raw/design_moving/exp20230524-000
```

输入应指定到 `exp...` 轨迹，不是建筑目录。相对输入路径以项目根目录为基准。
数据类型和输入路径不能在算法 YAML 中重复设置；可用 `--data-config` 指定另一份
数据配置，或用原 `--dataset-type`、`--in-traj-folder` 参数临时覆盖这两项。

唯一算法配置为 [config/imaging.yaml](config/imaging.yaml)。命令均在项目根目录运行：

```bash
# 默认执行重建、预测后激光评价和三幅中文图
python -m code.image_scene
# 查看数据选择和计算规模，不读取传感器数据
python -m code.image_scene --dry-run
# 显式指定配置、目标圈及后端
python -m code.image_scene --config code/config/imaging.yaml --frame-num 2 --backend cpu
# 固定平台，雷达仍随电机旋转
python -m code.image_scene --dataset-type PanoData --platform-motion static \
  --in-traj-folder /mnt/dataset/panoradar/raw/DRL_static/exp20230507-000 --frame-num 0
# 从已保存的统一预测重新评价并补齐缺图，不重新推理
python -m code.image_scene --evaluate-only /path/to/scene_00002.npz
# 无图运行仍保存固定预测与评价
python -m code.image_scene --no-plot
```

旧 `--evaluate` 参数兼容保留；现在默认已经评价，不需要该参数。旧版仅包含表面、
没有原方法基线与回波法线的 NPZ 无法生成新的三幅图，需用统一入口重新计算。
`--evaluate-only` 使用预测报告保存的设置，不按当前 `data.yaml` 更换数据或评价口径。

默认目标圈为 2，使用居中的五圈窗口，边界截断，表面优化默认执行 80 次。
`platform_motion` 由配置明确选择 `static/moving`，不按目录名猜测。固定平台采用
固定平台位姿，雷达仍随电机旋转。场景目标固定在世界坐标中，暂不处理动态目标、
身份关联、跨帧遮挡推断或全局地图融合。

## 三幅图与输出文件

默认目录为
`outputs/imaging/<场景>/<轨迹>/<模式>_unified_<时间戳>/`，拒绝覆盖已有圈产物。
第 2 圈的三幅 PNG 为：

| 文件 | 内容 | 误差口径 |
|---|---|---|
| `rf_range_comparison_00002.png` | 参照论文 Figure 9：原方法距离峰值投影、当前最终距离、激光距离 | 两种 RF 方法各自与同一激光参考的绝对距离误差及完整累计分布 |
| `normal_comparison_00002.png` | 当前回波法线、主激光法线、未排除玻璃的激光法线 | 仅计算当前法线与主激光参考的有方向夹角误差及累计分布；未排除玻璃的参考仅展示；不生成无权重的原方法神经网络法线 |
| `geometry_comparison_00002.png` | 参照论文 Figure 8：原方法、当前方法、激光各自的三维点云及零高度带俯视图 | 同一激光角单元对应点的三维位置距离及累计分布 |

Figure 9 的“二维热图”是每个方向选择距离峰值后的全景距离投影，不是 RF 幅度
与 LiDAR 距离直接相减。PanoData 全景为 `64×512`，SelfData 为 `64×1024`。
两者俯仰覆盖 ±45°、方位覆盖 360°，最后一列重复第一列。坐标轴直接使用角度，
64 行搜索网格不等于实际俯仰分辨率或天线
有效视场。没有激光覆盖的区域不补造参考值。

SelfData 的全景距离、法线和对应误差图仅显示俯仰 −26°～4°，PanoData 保持
−45°～45°。显示按原角度坐标裁剪，不拉伸、重映射或补全数据；预测、激光参考、
评价指标及累计分布仍使用完整评价区域，三维点云与零高度带俯视图保留原范围。

Figure 8 风格的零高度带俯视图直接从相同的三维点云取
`abs(z) <= lidar_slice_tolerance_m`，默认 ±0.1 米，不再独立运行水平 SAR。
它是高度带，不是精确 `z=0` 截面。三维位置误差采用固定角单元的对应点，而不是
根据预测误差挑选最近激光邻居。RF 点云保留全部候选，不按置信度删点。

同目录还保存：

- `scene_00002.npz`：RF-only 固定预测，包括当前/原方法距离与点、回波法线、
  支持、相干性、反射估计、局部曲面参数、逐采样位姿和原始轨迹。
- `scene_00002.ply`：当前统一表面的点、强度和支持标记，排除重复方位列。
- `report_00002.json`：格式、配置、源码/输入指纹、预测 SHA256、采样区间、
  RF 验收历史、法线来源、耗时、限制及产物清单。
- `evaluation_00002.json`：原方法/当前方法的距离及位置指标、当前法线与主激光参考的
  指标、有效数量、同步偏差和玻璃标注状态；不计算未剔除玻璃参考的法线误差。
- `lidar_reference_00002.npz`：预测后生成的两种激光参考、固定评价掩码及误差数组。
  它不被任何推理路径读取。

`--evaluate-only` 检查 RF 预测哈希；已有评价必须与重新计算的结果完全一致，才会
补齐缺失图片，不覆盖已有图片。解释文字、坐标轴、图例和单位使用中文；需要环境中
已提供可用中文字体，无字体时明确报错。依赖在根目录 `requirements.txt`；CUDA
另需环境中已有匹配版本的 CuPy。

## 唯一 RF 重建及原方法基线

### 原方法基线

使用 PanoRadar 论文公式 (11) 的远场波束形成及距离 FFT，在目标圈逐方向选择配置
距离区间内的最强距离峰，得到论文 Figure 9 所定义的投影。复用目标圈成像体中的
原生 FFT 格点，去掉补零带来的额外采样。基线和当前方法使用同一输入轨迹、数据
半径、波形、通道约定和孔径配置。

这是本地实现的无学习模型信号处理基线，不是官方学习模型或 CFAR 点云提取器的
逐字节复现。基线使用目标圈；当前方法按配置使用邻圈。图8风格的原方法点云来自
图9中的同一组距离峰，不将当前方法的初始化冒充原方法。

### 当前表面与轨迹

1. 逐 chirp 计算电机相位和平台位姿，把真实天线位置变换到目标圈开始时 RF 坐标系。
   接缝仅使用实际邻接采样，不复制同一 chirp 充当另一位置。
2. 目标圈在配置距离范围内以 4.5 m 为界，每个方向在两侧各保留默认最多 5 个 RF
   距离峰，在各峰附近进行 ±7.5 cm、9 点的近场匹配功率搜索，并验证二次细化的顶点。
   各圈的 RF 幅度体配准到目标圈的原生距离格点。通过相对幅度和噪声底门限的响应
   累加归一化功率。每条射线在 4.5 m 内和以外各可补入一个有多圈支持、与已有峰
   分离的候选，单圈运行时跳过补峰。
3. 在各源圈自身坐标系中复查同一空间候选的 RF 幅度支持。每条射线优先使用有 RF
   支持的 4.5 m 内候选，缺少此类候选时才回退到更远的受支持候选。全部缺少 RF
   支持时保留有限距离供输出，标记 `surface_valid=false`。多圈运行时，从目标圈距离峰中
   取同时通过单峰噪声、近场相对幅度和跨圈支持门限的最近峰，且该峰的融合代价需在
   最低候选代价的 3 dB 以内。若有这样的峰，超出该峰一个距离容差的后方候选不参与
   选面。容差沿用跨圈补峰的最小间距。其余候选按融合代价选面。为限制内存，多圈
   提名和候选复查分别逐圈生成幅度体。全景四邻接图在跳变两侧的
   候选均满足 RF 支持条件、且各自与同侧邻点的首选深度相符时，降低该候选组合的跨边界
   代价。其余组合沿用截断距离惩罚。默认 12 次阻尼 min-sum BP，求解为近似解。
4. 以选面结果初始化正距离场和局部分片平面，使用默认 3 行 × 6 列的块，优化 RF
   匹配功率及几何/距离先验。选用受支持近峰的射线在微调时也以 4.5 m 为上限。
   两类移动数据均固定使用从 `motion_output.npz` 积分得到的位姿。默认距离改变量
   上限 7.5 cm，电机相位保持名义旋转。
5. 每轮遍历全部 RF 支持点，按实际批量加权积累梯度后更新一次 Adam，正则每轮
   只累积一次。`batch_size` 控制内存，不抽掉尾批或减少点覆盖。
6. 固定交替采样分为梯度与 RF 验收子集，但初始化使用了全部采样，因此验收子集
   不是独立测试集。按 RF 目标和全部支持点的对数功率增益验收；失败退回初始化。

局部平面法向只用于几何约束，不是输出的回波法线。模型是每方向单表面的局部
径向优化，不保证找回初始化之外的表面、多层遮挡、消除多径或达到全局最优。
RF 验收通过不意味着激光几何误差改善。

### 最终表面上的回波法线

优化结果固定后，在相同最终三维点和同一套天线几何上，复用相干匹配响应
计算 mmNorm 带正负号的方向投票。使用与优化相同的真实固定孔径和视场掩码，不
重新选面，不运行第二套轨迹，不从距离图求空间梯度，也不把局部平面法向作为预测。

默认 `temporal_fusion: multilook` 在圈内相干，跨圈累加方向统计后归一化。
`coherent` 适用于跨圈相位稳定的固定平台。未定义法线以零 RHS 表示，主角度评分
为 90°。

内部点到天线方向固定翻转为原标签的朝外约定；训练 XYZ 与 RF XYZ 转换为
`diag(1,-1,-1)`，再按固定全景基底转换为 `pred_rhs`，不根据标签选择符号。

方位列数 `W` 在 PanoData 中为 512，在 SelfData 中为 1024。

| 固定预测字段 | 形状 |
|---|---|
| `predicted_range_m`、`panoradar_range_m` | `(64,W)` |
| `points_m`、`panoradar_points_m`、`world_points_m` | `(64,W,3)` |
| `pred_rhs` | `(3,64,W)` |
| `normal_comparison` | `(64,W,3)`，RF 坐标系朝外法线 |
| `normal_defined`、`normal_valid`、`surface_valid` | `(64,W)`，标记不删除主评分像素 |
| `world_from_platform`、`initial_world_from_platform` | 同一组逐实际采样位姿，后者用于兼容旧产物格式 |

世界坐标依赖输入或积分轨迹的基准，不自动等于地理坐标。

## PanoData 与 SelfData

RF 输入均为 2 Tx × 4 Rx 的复数数据，展开成 8 通道，默认扣除整条轨迹逐通道复数
均值后除以 32768。输入只读，输出禁止位于两类数据根目录或输入轨迹内，包含符号
链接真实路径检查；不在数据目录写缓存。

| 数据 | RF 与时钟 | 波形/位姿 |
|---|---|---|
| PanoData | `radar_data.npy`、`radar_ts.txt`、`data_params.npz:radar_sep/r_radar` | 原 77–81 GHz 约定，保留原 5.5° 方向修正 |
| SelfData | 未压缩 `radar.npz:frames/timestamps`、`radar_config.yaml`、`data_params.npz:radar_sep/r_radar/imu_sep` | 读取实际参数；用户确认采用 IMU 分帧；RF 方向归约后平滑，不加 5.5° |

移动 PanoData 和 SelfData 均要求序列目录内有 `motion_output.npz`。文件缺失时，
异常会给出当前序列的离线生成命令。从文件仅读取 `v_esti/theta_v_esti/omega_imu`。
以每圈速度近似恒定，按雷达时间戳积分得到 SE(2) 相对位姿。位姿平移采用线性插值，
旋转采用 SLERP，查询须在时间覆盖范围内。位姿原点为旋转中心，轴为 RF 的 x 前、
y 左、z 上。
成像全程使用该轨迹，输入运动误差和积分漂移会反映在位姿中。

SelfData 对 IMU z 角速度按原规则异常处理和平滑，通过
`dtheta = 2π/N - delta_t*omega` 确定成像区间，末圈截断到有位姿覆盖的边界。
电机相位保持平台坐标中的名义旋转，平台 yaw 在逐 chirp 位姿中独立计入，避免
重复补偿。各圈投票核心与原 `radar_sep` 区间相交，互不重叠；报告原始、IMU
和核心采样区间。`positions` 是度制累计编码器角，但当前不启用 encoder 分帧。

SelfData 的 RF 成员直接只读内存映射，不解包或整体读入数 GB 数据；压缩、加密、
对象类型的 RF 成员明确报错。距离 FFT 等效带宽为 `slope*N/fs`，空间补偿保留
上游 `linspace(start_freq,start_freq+slope*ramp_end_time,N)`；示例分别约为
3.491607 GHz、3.90078 GHz，原生距离采样约 4.293 cm。空间网格沿用上游处理约定，
未加入 ADC 起始延迟，不将其宣称为重新标定的物理波形。

默认等效阵列相位中心为
`p[j,a]=[r*cos(theta[j]),r*sin(theta[j]),-s[a]/2]`。PanoData 的
`s[a]=1.9 mm*[7,6,5,4,3,2,1,0]`，SelfData 的间距取 `c/(2*start_freq)`；双程
路径为 `2*norm(x-p)`，不等于已标定的实际物理阵列。可选几何 JSON 的
`tx_offsets_m/rx_offsets_m` 为有限 `(8,3)` 数组、单位米、按 `(2 Tx,4 Rx)` C 顺序
排列；提供时分别变换 Tx/Rx，使用实际双站路径与方向平分线。

## 预测后激光评价与玻璃

从开始推理到 RF 距离、点、轨迹、法线与原方法基线全部固定并保存 NPZ/PLY/报告
为止，不加载、查询或使用 LiDAR、相机及其逐样本派生信息。所有 GT 参考、玻璃
掩码、误差、图的色标和可视化都在此后生成，不改变、选择或校准任何 RF 预测。

PanoData 按原始激光重建距离图，与处理后深度做唯一精确匹配，读取同名
`glass_npy`；缺失或无法唯一匹配时失败。主法线参考保留原 `depth[glass]=-1000`、
梯度、滤波、RGB 量化和线缆掩码规则；未剔除玻璃的参考从同一原始深度重新计算，
而不是仅把主参考的白色区域补色。通过玻璃后的激光回波不是玻璃表面真值。
主法线评分仍采用原 `SnEvaluator` 的掩码与余弦夹角、均值、中位数、P80、P90。

SelfData 仅读取同步的 `point_cloud_raw.npz`、`lidar_sep/transform`；使用
`lidar_sep[frame_num]` 的扫描，报告其与 RF 圈起点的时间偏差。坐标为行向量
`XYZ @ block_diag(transform.T,1) + lidar_translation_m`，默认平移 `[0,0.06,-0.4]` 米；
按用户确认的安装关系，激光位于雷达下方约 0.4 米，RF z 轴向上，因此竖直平移为负。
水平偏移保留原配置，可在算法 YAML 中替换。对象类型扫描集合的反序列化仅发生在
评价阶段，可能需要整条轨迹点云的内存，不进入推理。

点云按已保存预测的方位宽度投影。新 SelfData 预测有 `64×1023` 个独立角单元，
历史 512 列预测按 `64×511` 个独立角单元评价。每格取最近实测距离，不补洞或插值。法线使用
实际点云局部 PCA，默认半径 0.25 米、最多 32 个邻居、至少 6 个且有两个独立
切向维度，取最小特征值方向并统一朝外；不再按平面度/曲率选取评分点。局部
边缘、稀疏性、扫描同步和未做逐点运动去畸变都会影响参考精度。

SelfData 没有像素玻璃标签，不能从物体框或 RF 结果猜玻璃掩码。因此两种激光法线
相同，图和报告明确标注这一事实。它是局部 PCA 参考，不是原 PanoData 训练标签，
两类数据的法线指标不能直接当作同一基准。

距离和三维位置误差使用相同 GT-only 有效距离/位置掩码，保留玻璃区域，仅排除重复
全景端点；法线仅按主激光参考的有效掩码评分，未剔除玻璃的激光法线仅供展示。
RF 支持、置信度、误差大小或最近邻距离均不筛选主评分像素。

所有误差热图使用相同颜色含义：近黑色表示误差小，浅黄色表示误差大，白色表示
无有效评价值；图中保留数值色标，并标注中文说明。距离和三维位置误差的单位为米，
法线夹角误差的单位为度。

## 目录与验证

```text
code/
├── config/          # data.yaml、imaging.yaml、统一默认值和评价设置
├── dataset_io/      # PanoData/SelfData，只读输入和输出目录保护
├── motion/          # 平台位姿、真实孔径与跨圈方向统计
├── imaging/         # 波形、波束形成、选面、联合优化、唯一运行入口
├── normals/pixels.py # 固定法线坐标转换与全景打包
├── evaluation/      # 预测后两类激光参考、角度/距离/位置指标
├── visualization/   # 三幅中文对比图，以及独立激光查看工具
├── image_scene.py   # 唯一成像命令
├── visualize_lidar.py # 独立离线激光查看，不是另一套 RF 重建
└── README.md
```

根包保留标准库 `code` 的交互解释器接口，供调试器使用；只加载当前 Python 的
标准库，不引用参考目录。旧 `code/tests/` 已按要求删除，未创建替代测试脚本。
可运行已有入口与静态检查：

```bash
ruff check --target-version py310 code
python -m code.image_scene --dry-run
python -m code.image_scene --help
python -m code.visualize_lidar --help
```

完整默认配置的耗时、GPU 行为和整轨迹精度仍须实测，不能用小孔径、单次迭代的
入口核验结果代替。激光参考、标定和局部优化都有上述限制。

## 方法来源

- [Enabling Visual Recognition at Radio Frequency / PanoRadar](https://arxiv.org/html/2405.19516v1)：
  公式 (11)、波形与符号、Figure 9 的距离峰值投影、Figure 8 的三维与二维展示。
  未加载其分辨率增强模型；8 个等效通道和 64 行搜索不代表 64 个独立俯仰分辨单元。
- [mmNorm 论文](https://laura-dodds.github.io/papers/mmNorm.pdf)、官方
  [image_processor.py](https://github.com/signalkinetics/mmNorm/blob/main/src/data_processing/image_processor.py)、
  [法线内核](https://github.com/signalkinetics/mmNorm/blob/main/src/data_processing/cuda/surface_normal_est.cu)：
  逐观测路径补偿与带符号方向投票。多峰与多视融合是本项目的工程适配。
- [多视立体的候选与空间一致性思路](https://www.microsoft.com/en-us/research/?p=610152)：
  本项目使用 RF 幅度支持和截断图代价，不是该光学算法的原样实现。
- [PatchMatch Stereo: Stereo Matching with Slanted Support Windows](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/PatchMatchStereo_BMVC2011_6MB.pdf)：
  深度边界处的自适应邻域支持。本项目用多圈 RF 证据和同侧深度连续性调整图代价。
- [Enhancing SAR Multipath Ghost Image Suppression for Complex Structures through Multi-Aspect Observation](https://www.mdpi.com/2072-4292/16/4/637)：
  多视角对照中的散射位置一致性。本项目用空间配准后的 RF 功率支持提名候选，仍可能保留跨视角稳定的多径。
- [可微 SAR 与自动聚焦](https://ieeexplore.ieee.org/document/9455345)为 RF 匹配
  功率优化提供参考。
- SelfData 规则参考本机 [MyRadar 自采几何](/data/glz/project/MyRadar/tools/suradar_core.py)、
  [SURadar 数据与 IMU 分帧](/data/glz/project/SURadar/suradar_SP/dataset.py)、
  [RF 运动估计](/data/glz/project/SURadar/suradar_SP/motion_estimation.py)、
  [装置外参](/data/glz/project/SURadar/model/config.py)。仅参考规则，运行时不导入这些仓库。
