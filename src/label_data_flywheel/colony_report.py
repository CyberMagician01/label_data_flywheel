"""服务器绘制可导出科学图与离线浏览页；私有视频底图不进入代码仓库。"""

from pathlib import Path
import base64
import html
import json
import numpy as np
from .io import write_json


def render_report(report, output, background=None, font_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from PIL import Image

    if font_path:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=font_path
        ).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for si, scope in enumerate(report["scopes"]):
        windows = scope["windows"]
        t = [(w["start_seconds"] + w["end_seconds"]) / 2 for w in windows]
        if scope["unique_track_ids"] == 0:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
            a = axes[0]
            a.plot(t, [w["mean_observed_count"] for w in windows], "o-", label="直接观测")
            count_changes = [c for c in scope["temporal"]["change_candidates"] if c["metric"] == "mean_observed_count"]
            if count_changes:
                lookup = {w["index"]: ti for w, ti in zip(windows, t)}
                a.scatter([lookup[c["window_index"]] for c in count_changes], [c["value"] for c in count_changes], color="#d25432", s=65, zorder=3, label="变化复核候选")
            a.set(title="每窗平均观测数", ylabel="只 / 帧", xlabel="源视频时间 / 秒")
            a.legend()
            a.grid(alpha=0.2)
            a = axes[1]
            a.plot(t, [w["density_cv"] for w in windows], "o-", color="#398575")
            a.set(title="空间分布不均匀程度", ylabel="密度变异系数", xlabel="源视频时间 / 秒")
            a.grid(alpha=0.2)
            a = axes[2]
            im = a.imshow(scope["mean_density_map"], cmap="magma", extent=[0, 1, 1, 0], aspect="auto")
            a.set(title="时均空间密度", xlabel="归一化 x", ylabel="归一化 y")
            fig.colorbar(im, ax=a, label="平均观测目标数 / 格")
        else:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
            a = axes[0, 0]
            a.plot(t, [w["mean_observed_count"] for w in windows], "o-", label="实际观测")
            a.plot(
                t,
                [w["mean_visible_interpolations"] for w in windows],
                "o-",
                label="可见插值（单列）",
            )
            a.set(title="每窗平均目标数", ylabel="只 / 帧")
            a.legend()
            a = axes[0, 1]
            a.plot(
                t,
                [w["active_fraction"] for w in windows],
                "o-",
                label="活动比例（可测运动者中）",
            )
            a.plot(
                t,
                [w["motion_observable_fraction"] for w in windows],
                "--",
                label="运动可测比例",
            )
            a.set(title="活动及可观测性", ylim=(0, 1.05))
            a.legend()
            a = axes[1, 0]
            im = a.imshow(
                scope["mean_density_map"], cmap="magma", extent=[0, 1, 1, 0], aspect="auto"
            )
            a.set(title="时均空间密度（观测框中心）", xlabel="归一化 x", ylabel="归一化 y")
            fig.colorbar(im, ax=a, label="平均观测目标数 / 格")
            # 展示中间窗的完整图，不只抽取最密或最有利区域。
            net = windows[len(windows) // 2]["network"]
            a = axes[1, 1]
            positions = {n["id"]: n["position_normalized"] for n in net["nodes"]}
            lines = [[positions[e["source"]], positions[e["target"]]] for e in net["edges"]]
            if lines:
                weights = [e["qualified_proximity_seconds"] for e in net["edges"]]
                a.add_collection(
                    LineCollection(
                        lines,
                        array=np.asarray(weights),
                        cmap="viridis",
                        linewidths=0.5,
                        alpha=0.6,
                    )
                )
            if positions:
                xy = np.asarray(list(positions.values()))
                a.scatter(xy[:, 0], xy[:, 1], s=7, c="#e68a23")
            a.set(
                xlim=(0, 1),
                ylim=(1, 0),
                title=f"中间窗近邻网络：{len(positions)} 个累计轨迹节点 / {len(lines)} 条边",
                xlabel="归一化 x",
                ylabel="归一化 y",
            )
            a = axes[2, 0]
            a.plot(
                t, [w["median_speed_bl_proxy_s"] for w in windows], "o-", label="中位速度"
            )
            a.set(
                title=f"运动时序（{report['config']['motion_lag_seconds']:g} 秒位移窗，拒绝明显跳变）",
                ylabel="身体长度代理 / 秒",
            )
            a = axes[2, 1]
            if scope["gate_status"] == "verified_entrance":
                for direction, label in [("in", "向内侧"), ("out", "向外侧")]:
                    values = [
                        w["line_crossings"][direction] * 60 / w["observed_exposure_seconds"]
                        for w in windows
                    ]
                    a.plot(t, values, "o-", label=label)
                a.legend()
                a.set(
                    title="已标定巢口通量",
                    ylabel="观测穿越次数 / 分钟",
                )
            else:
                a.plot(t, [w["network"]["edge_density"] for w in windows], "o-")
                a.set(title="近邻网络随时间变化", ylabel="实际边 / 可选边")
            for a in [axes[0, 0], axes[0, 1], axes[2, 0], axes[2, 1]]:
                a.set_xlabel("源视频时间 / 秒")
                a.grid(alpha=0.2)
        lo, hi = scope["source_frame_range"]
        fig.suptitle(
            f"{scope['video']} | {scope['fps']:g} FPS | 源帧 {lo}–{hi}\n群体视频观测：个体数量 · 空间结构 · 短时变化",
            fontsize=16,
        )
        path = out / f"scope_{si}_overview.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        images.append((scope["video"] + "：群体统计总览", path))
        indices = sorted(set([0, len(windows) // 2, len(windows) - 1]))
        vmax = max(np.max(w["density_mean_objects_per_cell"]) for w in windows)
        fig, axs = plt.subplots(
            1,
            len(indices),
            figsize=(5 * len(indices), 4),
            squeeze=False,
            constrained_layout=True,
        )
        for a, wi in zip(axs[0], indices):
            w = windows[wi]
            im = a.imshow(
                w["density_mean_objects_per_cell"],
                extent=[0, 1, 1, 0],
                cmap="magma",
                vmin=0,
                vmax=vmax,
                aspect="auto",
            )
            a.set(
                title=f"{w['start_seconds']:.1f}–{w['end_seconds']:.1f} 秒\n平均 {w['mean_observed_count']:.1f} 只",
                xlabel="归一化 x",
                ylabel="归一化 y",
            )
        fig.colorbar(im, ax=axs[0].tolist(), label="平均观测目标数 / 格（统一色标）")
        path = out / f"scope_{si}_density_windows.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        images.append((scope["video"] + "：不同时段密度对照", path))
        if background and len(report["scopes"]) == 1:
            fig, a = plt.subplots(figsize=(13, 8), constrained_layout=True)
            a.imshow(Image.open(background), extent=[0, 1, 1, 0])
            a.imshow(
                scope["mean_density_map"],
                extent=[0, 1, 1, 0],
                alpha=0.45,
                cmap="magma",
                aspect="auto",
            )
            entrance = report["config"].get("entrance") if scope["gate_status"] == "verified_entrance" else None
            if entrance:
                line = np.asarray(entrance["line"])
                a.plot(line[:, 0], line[:, 1], color="#00ffff", linewidth=3)
            a.set(
                title=f"{scope['video']}：原画面与时均密度位置对应\n"
                + ("青色线为配置的过线位置；" if entrance else "")
                + "热图不是分割掩膜",
                xlim=(0, 1),
                ylim=(1, 0),
            )
            path = out / f"scope_{si}_spatial_overlay.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            images.append((scope["video"] + "：空间位置核对", path))
    interpretation = report.get("apiculture", {})
    cards = interpretation.get("literature", {}).get("cards", [])
    cards_html = "".join(
        f"<li><b>{html.escape(c['title'])}</b>：{html.escape(c['evidence'])} <a href='{html.escape(c['url'], quote=True)}'>研究原文</a></li>"
        for c in cards
    )
    figures = "".join(
        f"<section><h2>{html.escape(label)}</h2><img src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'></section>"
        for label, path in images
    )
    statuses = {}
    for row in interpretation.get("interpretations", []):
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    def number(value):
        return "—" if value is None else f"{value:.4g}"

    finding_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            r["video"], "–".join(map(str, r["source_frame_range"])),
            r["title"], number(r["value"]), number(r["baseline_median"]),
            number(r["robust_z"]), "变化复核" if r["status"] == "visual_change" else "先核对可测条件",
        )) + "</tr>"
        for r in interpretation.get("interpretations", [])
    )
    findings_html = (
        "<table><thead><tr><th>视频</th><th>源帧区段</th><th>变化</th><th>当前值</th><th>历史基线</th><th>稳健偏离</th><th>复核方向</th></tr></thead><tbody>"
        + finding_rows + "</tbody></table>"
        if finding_rows else "<p>当前区段没有达到所设阈值的变化候选。</p>"
    )
    identity_note = "沿用输入标注的 ID；累计轨迹节点不等于同一帧的蜜蜂数量。"
    if any(
        r["input"].get("tracking_mode") == "analysis_baseline"
        for r in report.get("source_receipts", [])
    ):
        identity_note = "本报告使用分析用备选关联 ID，未替换原标注；不是原版 SAM 效果报告。短轨迹较多，运动和网络仅作为可观测候选。累计轨迹节点不等于同一帧蜜蜂数量。"
    if all(s["unique_track_ids"] == 0 for s in report["scopes"]):
        identity_note = "本次输入为已有检测与姿态标注，未提供持续 ID；结果展示数量、空间密度及其变化。"
    # 原始结构可在浏览页折叠查看；图片内嵌，可离线搬运。
    text = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>蜂群行为与数据飞轮</title>
    <style>body{{font:16px/1.8 system-ui;background:#f4f5f7;color:#203045;max-width:1200px;margin:30px auto;padding:20px}}section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}img{{width:100%}}li{{margin:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3e9}}</style>
    <h1>蜂群视频观测与变化复核</h1><p>依据源视频中的直接观测，分析个体数量、运动、空间密度与持续近邻关系；插值数量单独列出。</p>
    <p>{html.escape(identity_note)}</p>
    <section><h2>视频变化复核</h2>{findings_html}</section>
    {figures}<section><h2>视觉表征的研究依据</h2><ul>{cards_html}</ul></section>
    <details><summary>展开源帧与测量依据</summary><pre>{html.escape(json.dumps(interpretation.get("interpretations", []), ensure_ascii=False, indent=2))}</pre></details></html>"""
    (out / "index.html").write_text(text, encoding="utf-8")
    write_json(
        out / "render_manifest.json",
        {
            "images": [p.name for _, p in images],
            "private_background_included": bool(background),
        },
    )
    return [str(p) for _, p in images]
