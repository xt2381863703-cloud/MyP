"""Three Chinese comparison figures from one immutable RF reconstruction."""

import numpy as np

from ..imaging.coordinates import normal_color_mapping, rhs_to_normal

ERROR_COLOR_NOTE = (
    "误差图颜色：近黑色表示误差小，浅黄色表示误差大；白色表示无有效评价值"
)


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (
            name
            for name in (
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "WenQuanYi Zen Hei",
                "Microsoft YaHei",
                "SimHei",
            )
            if name in available
        ),
        None,
    )
    if family is None:
        raise RuntimeError("未找到可用的中文字体，无法生成中文解释文字")
    return plt, {"font.family": [family], "axes.unicode_minus": False, "font.size": 11}


def _elevation_limits(args):
    return (-26, 4) if args.dataset_type == "SelfData" else (-45, 45)


def _panorama(ax, values, title, cmap, low, high, args):
    image = ax.imshow(
        np.ma.masked_invalid(values),
        origin="upper",
        extent=(0, 360, -45, 45),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=low,
        vmax=high,
    )
    ax.set(
        title=title,
        xlabel="顺时针方位角（度）",
        ylabel="俯仰角（度）",
        ylim=_elevation_limits(args),
    )
    return image


def _describe(stats, unit):
    if stats["count"] == 0:
        return "无有效激光参考"
    return f"平均 {stats['mean']:.3f}{unit}；中位数 {stats['median']:.3f}{unit}；90%分位 {stats['p90']:.3f}{unit}；{stats['count']}个像素"


def _error_limit(arrays):
    maxima = [
        float(value[np.isfinite(value)].max())
        for value in arrays
        if np.isfinite(value).any()
    ]
    return max([1e-6, *maxima])


def _cdf(ax, entries, unit, title):
    drawn = False
    for values, label, color in entries:
        ordered = np.sort(values[np.isfinite(values)])
        if len(ordered):
            ax.step(
                ordered,
                np.arange(1, len(ordered) + 1) / len(ordered),
                where="post",
                label=f"{label}（{len(ordered)}个像素）",
                color=color,
            )
            drawn = True
    if drawn:
        ax.legend(loc="lower right")
    else:
        ax.text(
            0.5, 0.5, "无有效激光参考", ha="center", va="center", transform=ax.transAxes
        )
    ax.set(
        title=title,
        xlabel=f"误差（{unit}）",
        ylabel="累计像素比例",
        xlim=(0, None),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)


def plot_ranges(prediction, reference, evaluation, errors, metadata, args, output):
    plt, style = _plotting()
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            6,
            1,
            figsize=(21, 19),
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1.6]},
            layout="constrained",
        )
        fig.suptitle(
            f"第 {args.frame_num} 圈｜全景距离投影对比（参照论文图9）\n原方法为无学习模型的波束形成距离峰值；当前方法为统一优化表面\n两种误差使用相同的激光有效像素，不按射频置信度删点\n全景图显示俯仰 {_elevation_limits(args)} 度；统计与累计分布使用完整评价区域\n{ERROR_COLOR_NOTE}",
            fontsize=13,
        )
        depth = np.where(errors["range_mask"], reference["range_image_m"], np.nan)
        for ax, values, title in (
            (
                axes[0],
                prediction["panoradar_range_m"],
                "PanoRadar 原方法：三维射频热图的距离峰值投影",
            ),
            (
                axes[1],
                prediction["predicted_range_m"],
                "当前方法：同一套最终表面的全景距离",
            ),
            (axes[2], depth, "激光：相同角网格上的实测距离"),
        ):
            image = _panorama(
                ax, values, title, "magma_r", args.min_range, args.max_range, args
            )
            fig.colorbar(image, ax=ax, label="距离（米）")
        limit = _error_limit(
            [errors["range_error_panoradar"], errors["range_error_current"]]
        )
        for ax, name, label in (
            (axes[3], "panoradar", "原方法"),
            (axes[4], "current", "当前方法"),
        ):
            image = _panorama(
                ax,
                errors[f"range_error_{name}"],
                f"{label}与激光的绝对距离误差｜{_describe(evaluation['range'][name], '米')}",
                "magma",
                0,
                limit,
                args,
            )
            fig.colorbar(
                image, ax=ax, label="绝对误差（米）\n近黑色误差小，浅黄色误差大"
            )
        _cdf(
            axes[5],
            [
                (errors["range_error_panoradar"], "原方法与激光", "darkorange"),
                (errors["range_error_current"], "当前方法与激光", "steelblue"),
            ],
            "米",
            "完整距离误差累计分布（包含全部有效评价像素）",
        )
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_normals(
    prediction, reference, unmasked, evaluation, errors, metadata, args, output
):
    plt, style = _plotting()
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(21, 17),
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.6]},
            layout="constrained",
        )
        if metadata["glass_labels_available"]:
            note = f"已提供玻璃掩码，共 {metadata['glass_pixels']} 个像素；保留玻璃区域的激光回波不等于玻璃表面真值"
            filtered_title = "激光法线：按原评价规则剔除玻璃区域"
        else:
            note = "自采数据未提供玻璃像素掩码：两种激光参考相同，不推断玻璃区域，也不补造缺失法线"
            filtered_title = "激光法线：当前评价参考（未提供玻璃标注，未作玻璃剔除）"
        fig.suptitle(
            f"第 {args.frame_num} 圈｜回波法线对比\n当前法线由最终表面和同一校正轨迹上的复数回波投票计算；不使用距离梯度\n{note}\n保持原法线颜色和有方向夹角计算；未定义的射频法线按90度计分；未剔除玻璃的参考仅展示\n全景图显示俯仰 {_elevation_limits(args)} 度；统计与累计分布使用完整评价区域\n{ERROR_COLOR_NOTE}",
            fontsize=12,
        )
        current = rhs_to_normal(prediction["pred_rhs"].transpose(1, 2, 0))
        panels = [
            (
                current,
                errors["normal_mask_filtered"],
                "当前方法：回波法线（主评价区域）",
            ),
            (
                reference["training_normal_xyz"],
                errors["normal_mask_filtered"],
                filtered_title,
            ),
            (
                unmasked["training_normal_xyz"],
                unmasked["valid"].reshape(64, 512),
                "激光法线：保留玻璃区域的全部有效测量",
            ),
        ]
        for ax, (normal, mask, title) in zip(axes[:3], panels):
            with np.errstate(invalid="ignore", divide="ignore"):
                color = normal_color_mapping(normal.copy())
            color[~mask] = 255
            ax.imshow(
                color,
                origin="upper",
                extent=(0, 360, -45, 45),
                aspect="auto",
                interpolation="nearest",
            )
            ax.set(
                title=title,
                xlabel="顺时针方位角（度）",
                ylabel="俯仰角（度）",
                ylim=_elevation_limits(args),
            )
        metrics = evaluation["normal"]["filtered"]
        note = (
            "无有效参考"
            if metrics["sn_angle_mean"] is None
            else f"平均 {metrics['sn_angle_mean']:.2f}度；中位数 {metrics['sn_angle_median']:.2f}度；{metrics['evaluation_pixels']}个像素"
        )
        image = _panorama(
            axes[3],
            errors["normal_error_filtered"],
            f"当前方法与主激光参考的法线夹角误差｜{note}",
            "magma",
            0,
            180,
            args,
        )
        fig.colorbar(
            image, ax=axes[3], label="有方向夹角（度）\n近黑色误差小，浅黄色误差大"
        )
        _cdf(
            axes[4],
            [
                (errors["normal_error_filtered"], "当前方法与主激光参考", "steelblue"),
            ],
            "度",
            "当前方法与主激光参考的法线误差累计分布",
        )
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_geometry(prediction, reference, evaluation, errors, metadata, args, output):
    plt, style = _plotting()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    with plt.rc_context(style):
        fig = plt.figure(figsize=(20, 24), layout="constrained")
        grid = fig.add_gridspec(5, 2, height_ratios=[2, 2, 2, 0.9, 1.1])
        tolerance = args.lidar_slice_tolerance_m
        fig.suptitle(
            f"第 {args.frame_num} 圈｜俯仰结构与俯视点云对比（参照论文图8）\n左列为三维点云，右列为同一套点云在零高度附近的俯视图：|高度|≤{tolerance:g}米\n射频保留所有表面候选，不按置信度或激光误差删点；高度带不是精确平面截面\n全景图显示俯仰 {_elevation_limits(args)} 度；统计与累计分布使用完整评价区域\n{ERROR_COLOR_NOTE}",
            fontsize=13,
        )
        unique = np.ones((64, 512), dtype=bool)
        unique[:, -1] = False
        truth = reference["points_m"].reshape(64, 512, 3)
        clouds = [
            (prediction["panoradar_points_m"][unique], "PanoRadar 原方法距离峰值点云"),
            (prediction["points_m"][unique], "当前方法统一表面点云"),
            (truth[errors["position_mask"]], "激光实测点云（相同角网格与评价范围）"),
        ]
        zvalues = [points[:, 2] for points, _ in clouds if len(points)]
        low = min((float(z.min()) for z in zvalues), default=-1)
        high = max((float(z.max()) for z in zvalues), default=1)
        if high <= low:
            high = low + 1
        norm = Normalize(low, high)
        extent = args.max_range
        for row, (points, name) in enumerate(clouds):
            ax = fig.add_subplot(grid[row, 0], projection="3d")
            if len(points):
                ax.scatter(
                    *points.T,
                    c=points[:, 2],
                    cmap="turbo",
                    norm=norm,
                    s=1,
                    linewidths=0,
                    rasterized=True,
                )
            ax.scatter([0], [0], [0], marker="*", color="black", s=70)
            ax.set(
                title=f"{name}｜{len(points)}个点",
                xlabel="前向 x（米）",
                ylabel="横向 y（米）",
                zlabel="高度 z（米）",
                xlim=(-extent, extent),
                ylim=(-extent, extent),
                zlim=(low, high),
            )
            ax.set_box_aspect((2 * extent, 2 * extent, high - low))
            ax.view_init(elev=25, azim=-60)
            fig.colorbar(
                ScalarMappable(norm=norm, cmap="turbo"),
                ax=ax,
                shrink=0.65,
                label="高度（米）",
            )
            ax = fig.add_subplot(grid[row, 1])
            selected = points[np.abs(points[:, 2]) <= tolerance]
            if len(selected):
                ax.scatter(
                    selected[:, 0],
                    selected[:, 1],
                    s=4,
                    color="black",
                    linewidths=0,
                    rasterized=True,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "该高度带内没有点",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.scatter(
                [0], [0], marker="*", color="darkorange", s=100, label="雷达原点"
            )
            ax.set(
                title=f"{name}的零高度带俯视图｜{len(selected)}个点",
                xlabel="前向 x（米）",
                ylabel="横向 y（米）",
                xlim=(-extent, extent),
                ylim=(-extent, extent),
                aspect="equal",
            )
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
        limit = _error_limit(
            [errors["position_error_panoradar"], errors["position_error_current"]]
        )
        for column, name, label in (
            (0, "panoradar", "原方法"),
            (1, "current", "当前方法"),
        ):
            ax = fig.add_subplot(grid[3, column])
            image = _panorama(
                ax,
                errors[f"position_error_{name}"],
                f"{label}与激光的三维位置误差\n{_describe(evaluation['position'][name], '米')}",
                "magma",
                0,
                limit,
                args,
            )
            fig.colorbar(
                image, ax=ax, label="三维位置误差（米）\n近黑色误差小，浅黄色误差大"
            )
        ax = fig.add_subplot(grid[4, :])
        _cdf(
            ax,
            [
                (errors["position_error_panoradar"], "原方法与激光", "darkorange"),
                (errors["position_error_current"], "当前方法与激光", "steelblue"),
            ],
            "米",
            "同一激光像素对应点的三维位置误差累计分布",
        )
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_comparisons(
    prediction, reference, unmasked, evaluation, errors, metadata, args, outputs
):
    for function, path in zip((plot_ranges, plot_normals, plot_geometry), outputs):
        if path.exists():
            continue
        if function is plot_normals:
            function(
                prediction,
                reference,
                unmasked,
                evaluation,
                errors,
                metadata,
                args,
                path,
            )
        else:
            function(prediction, reference, evaluation, errors, metadata, args, path)
