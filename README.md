# 旋转毫米波雷达

唯一成像命令为 `python -m code.image_scene`：同一套最终表面和轨迹输出全景距离、
回波法线、三维与俯视结构三幅中文对比图，并进行预测后的激光评价。
数据类型与输入轨迹统一在 [data.yaml](code/config/data.yaml) 选择，算法参数位于
[imaging.yaml](code/config/imaging.yaml)，支持 PanoData 和采用 IMU 分帧的 SelfData。

输入输出、配置、玻璃掩码、误差口径、传感器隔离和参考资料见
[code/README.md](code/README.md)。旧的独立法线/水平成像入口及测试已移除。
`paper/` 保存参考资料，不参与运行；`pano/` 当前为空。依赖在 `requirements.txt`，
结果写入 `outputs/`。
