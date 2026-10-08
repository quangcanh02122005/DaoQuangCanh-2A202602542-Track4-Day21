"""Calibration QA cho LiDAR-camera (topic A): demo overlay, sweep drift, edge-score detector, latency.

Chạy từ gốc repo:
    python -m src.calib_qa --help
    python -m src.calib_qa demo                       # 3 overlay near/mid/far + ảnh lệch yaw
    python -m src.calib_qa sweep                      # bảng drift -> % FOV, % điểm trong 2D box
    python -m src.calib_qa edge                       # edge alignment score + ngưỡng phát hiện drift
    python -m src.calib_qa fail                       # ảnh fail_*.png + bảng lỗi đồng bộ thời gian
    python -m src.calib_qa latency                    # p50/p95 thời gian chiếu 1 frame
    python -m src.calib_qa all                        # chạy tất cả

Mọi phép tính đều tất định (không có random), chạy lại cho đúng cùng số.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.calib_metrics import (DIST_BINS, NEIGHBOURS, apply_drift, edge_score, finite_points,  # noqa: E402
                               image_edge_distance, in_box_hits, lidar_edge_points, object_point_sets)
from starter import nuscenes_io  # noqa: E402
from starter.datasets import list_frames, load_frame  # noqa: E402
from starter.projection import draw_box2d, overlay_points, project_velo_to_image  # noqa: E402

DATASETS = {"kitti": "data/kitti_mini", "nuscenes": "data/nuscenes_mini_subset"}
ROT_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]
TRANS_LEVELS = [0.0, 0.02, 0.05, 0.10]


def levels_for(kind: str) -> list[float]:
    return ROT_LEVELS if kind in ("yaw", "pitch", "roll") else TRANS_LEVELS


def put_label(img: np.ndarray, text: str, scale: float = 0.9) -> np.ndarray:
    """Chữ trắng trên dải nền đen ở góc trên-trái (dễ đọc kể cả khi ảnh bị thu nhỏ)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 16, h + base + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render(fr: dict, calib, title: str | None = None) -> np.ndarray:
    uv, depth, _ = project_velo_to_image(finite_points(fr["points"]), calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)
    return put_label(vis, title) if title else vis


# ---------------------------------------------------------------- demo

def cmd_demo(args) -> None:
    out = Path(args.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    # 3 khoảng cách: vật rất gần (<6 m), trung bình (người đi bộ ~10–25 m), xe xa (>50 m) — xem data/README.md
    for fid, tag in [("000019", "near"), ("000011", "mid"), ("000004", "far")]:
        fr = load_frame(DATASETS["kitti"], fid)
        p = out / f"demo_{tag}_kitti_{fid}.png"
        cv2.imwrite(str(p), render(fr, fr["calib"], f"KITTI {fid} ({tag}) - calib goc"))
        print("->", p)
    # cùng một frame, calib đúng vs yaw lệch 1° và 2°
    fr = load_frame(DATASETS["kitti"], "000011")
    rows = [render(fr, apply_drift(fr["calib"], "yaw", lv), f"yaw drift {lv:+.0f} deg") for lv in (0.0, 1.0, 2.0)]
    p = out / "demo_yaw_drift_kitti_000011.png"
    cv2.imwrite(str(p), np.vstack(rows))
    print("->", p)
    fr = load_frame(DATASETS["nuscenes"], "scene-0103_010")
    p = out / "demo_nuscenes_scene-0103_010.png"
    cv2.imwrite(str(p), render(fr, fr["calib"], "nuScenes scene-0103_010 - calib goc"))
    print("->", p)


# ---------------------------------------------------------------- sweep (pct_fov, pct_in_box)

def cmd_sweep(args) -> None:
    rows = []
    for ds in args.datasets:
        for fid in list_frames(DATASETS[ds]):
            fr = load_frame(DATASETS[ds], fid)
            pts, shape = finite_points(fr["points"]), fr["image"].shape
            objs = object_point_sets(pts, fr["calib"], fr["labels"], shape)
            for kind in args.kinds:
                for lv in levels_for(kind):
                    cal = apply_drift(fr["calib"], kind, lv)
                    _, _, m = project_velo_to_image(pts, cal, shape)
                    row = {"dataset": ds, "frame": fid, "kind": kind, "level": lv,
                           "n_points": len(pts), "n_fov": int(m.sum())}
                    hits = in_box_hits(pts, cal, objs, shape)
                    for lo, hi, name in DIST_BINS:
                        sel = [(h, n) for h, n, d in hits if lo <= d < hi]
                        row[f"hit_{name}"] = sum(h for h, _ in sel)
                        row[f"tot_{name}"] = sum(n for _, n in sel)
                        row[f"nobj_{name}"] = len(sel)
                    rows.append(row)
            print(f"[sweep] {ds} {fid}: {len(objs)} object có >=10 điểm")
    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "drift_sweep_frames.csv", index=False)

    agg = df.groupby(["dataset", "kind", "level"], sort=False).sum(numeric_only=True).reset_index()
    summ = pd.DataFrame({"dataset": agg.dataset, "kind": agg.kind, "level": agg.level,
                         "pct_fov": 100 * agg.n_fov / agg.n_points})
    hit_all = sum(agg[f"hit_{n}"] for *_, n in DIST_BINS)
    tot_all = sum(agg[f"tot_{n}"] for *_, n in DIST_BINS)
    summ["pct_in_box"] = 100 * hit_all / tot_all
    for *_, n in DIST_BINS:
        summ[f"pct_in_box_{n}"] = 100 * agg[f"hit_{n}"] / agg[f"tot_{n}"].replace(0, np.nan)
        summ[f"n_obj_{n}"] = agg[f"nobj_{n}"]
    base = summ[summ.level == 0].set_index(["dataset", "kind"])["pct_in_box"]
    summ["drop_in_box_pp"] = [base[(d, k)] - v for d, k, v in zip(summ.dataset, summ.kind, summ.pct_in_box)]
    summ = summ.round(2)
    summ.to_csv(out / "drift_sweep.csv", index=False)
    print(summ.to_string(index=False))
    plot_sweep(summ, out / "figures" / "drift_sweep.png")


def plot_sweep(summ: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    styles = {"kitti": "-", "nuscenes": "--"}
    for ds, g in summ.groupby("dataset"):
        for kind in ("yaw", "pitch", "roll"):
            s = g[g.kind == kind]
            ax[0].plot(s.level, s.pct_in_box, styles[ds], marker="o", label=f"{ds} {kind}")
        for kind in ("tx", "ty", "tz"):
            s = g[g.kind == kind]
            ax[1].plot(100 * s.level, s.pct_in_box, styles[ds], marker="o", label=f"{ds} {kind}")
        s = g[g.kind == "yaw"]
        for *_, n in DIST_BINS:
            ax[2].plot(s.level, s[f"pct_in_box_{n}"], styles[ds], marker="o", label=f"{ds} {n}")
    ax[0].set(xlabel="rotation drift (deg)", ylabel="% object points inside 2D box", title="Rotation drift")
    ax[1].set(xlabel="translation drift (cm)", ylabel="% object points inside 2D box", title="Translation drift")
    ax[2].set(xlabel="yaw drift (deg)", ylabel="% object points inside 2D box",
              title="Yaw drift by distance (<15 | 15-30 | >30 m)")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("->", path)


# ---------------------------------------------------------------- edge alignment detector

def ring_of(ds: str, fid: str, n_points: int) -> np.ndarray | None:
    """Cột ring (beam index) của nuScenes; starter.nuscenes_io bỏ cột này nên đọc lại file gốc."""
    if ds != "nuscenes":
        return None
    raw = np.fromfile(nuscenes_io.lidar_path(DATASETS[ds], fid), dtype=np.float32).reshape(-1, 5)
    assert len(raw) == n_points, "thứ tự điểm của load_frame khác file gốc"
    return raw[:, 4]


def cmd_edge(args) -> None:
    rows = []
    for ds in args.datasets:
        frames = list_frames(DATASETS[ds])
        for fid in frames:
            fr = load_frame(DATASETS[ds], fid)
            dist_map = image_edge_distance(fr["image"])
            exyz, ew = lidar_edge_points(fr["points"], ring_of(ds, fid, len(fr["points"])))
            for kind in args.kinds:
                for lv in levels_for(kind):
                    cal = apply_drift(fr["calib"], kind, lv)
                    row = {"dataset": ds, "frame": fid, "kind": kind, "level": lv,
                           "score": edge_score(exyz, ew, cal, dist_map)}
                    for nk, step in NEIGHBOURS:
                        row[f"nb_{nk}{step:+g}"] = edge_score(exyz, ew, apply_drift(cal, nk, step), dist_map)
                    rows.append(row)
            print(f"[edge] {ds} {fid}: {len(exyz)} LiDAR edge points")
    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    df.to_csv(out / "edge_score_frames.csv", index=False)

    analyze_edge(df, args)


# Hai cấu hình detector (so sánh B1): tìm trên cả 6 bậc tự do, hoặc chỉ tìm theo yaw.
GRIDS = {"all6dof": lambda c: c.startswith("nb_"), "yaw_only": lambda c: c.startswith("nb_yaw")}


def analyze_edge(df: pd.DataFrame, args) -> None:
    """Gộp theo cửa sổ W frame liên tiếp: nếu một calib trong lưới tìm kiếm quanh calib hiện tại có tổng
    score cao hơn, calib hiện tại không còn là cực đại -> nghi drift. drift_stat = best_nb / current - 1."""
    out = Path(args.out_dir)
    win = []
    for grid, pick in GRIDS.items():
        nb_cols = [c for c in df.columns if pick(c)]
        for ds, g in df.groupby("dataset", sort=False):
            order = {f: i // args.window for i, f in enumerate(list_frames(DATASETS[ds]))}
            g = g.assign(win=g.frame.map(order))
            for (kind, lv, w), gw in g.groupby(["kind", "level", "win"], sort=False):
                if len(gw) < args.window:
                    continue
                cur, nbs = gw.score.sum(), gw[nb_cols].sum()
                win.append({"grid": grid, "dataset": ds, "kind": kind, "level": lv, "window": w,
                            "first_frame": gw.frame.iloc[0], "score": cur / len(gw),
                            "best_neighbour": nbs.idxmax()[3:], "drift_stat": nbs.max() / cur - 1})
    wdf = pd.DataFrame(win)
    # Ngưỡng: lớn nhất của drift_stat khi calib ĐÚNG (level 0) + biên 0.005, riêng từng grid x dataset
    # (0 false alarm trên chính dữ liệu này — ngưỡng được chọn trên cùng tập, cần tập riêng khi triển khai)
    tau = wdf[wdf.level == 0].groupby(["grid", "dataset"]).drift_stat.max() + 0.005
    wdf["threshold"] = [tau[(gr, d)] for gr, d in zip(wdf.grid, wdf.dataset)]
    wdf["detected"] = wdf.drift_stat > wdf.threshold
    wdf.round(4).to_csv(out / "edge_score_windows.csv", index=False)
    summ = (wdf.groupby(["grid", "dataset", "kind", "level"], sort=False)
            .agg(mean_score=("score", "mean"), mean_drift_stat=("drift_stat", "mean"),
                 threshold=("threshold", "first"), detection_rate=("detected", "mean"),
                 n_windows=("window", "size")).reset_index().round(4))
    summ.to_csv(out / "edge_detection.csv", index=False)
    print(summ.to_string(index=False))
    plot_edge(summ, out / "figures" / "edge_score_vs_drift.png")


def plot_edge(summ: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    styles = {"kitti": "-", "nuscenes": "--"}
    for ax, (grid, gg) in zip(axes, summ.groupby("grid", sort=False)):
        for ds, g in gg.groupby("dataset"):
            for kind in ("yaw", "pitch", "roll", "tx", "ty", "tz"):
                s = g[g.kind == kind]
                x = s.level if kind in ("yaw", "pitch", "roll") else 10 * s.level  # 10 cm ~ "1" trên trục chung
                ax[0].plot(x, s.mean_score, styles[ds], marker="o", label=f"{ds} {kind}")
                ax[1].plot(x, s.mean_drift_stat, styles[ds], marker="o", label=f"{ds} {kind}")
                ax[2].plot(x, 100 * s.detection_rate, styles[ds], marker="o", label=f"{ds} {kind}")
            ax[1].axhline(g.threshold.iloc[0], color="k" if ds == "kitti" else "gray", ls=":",
                          label=f"threshold {ds} = {g.threshold.iloc[0]:.3f}")
        ax[0].set(title=f"[{grid}] Edge alignment score (mean / window)", ylabel="score")
        ax[1].set(title=f"[{grid}] drift_stat = best in search grid / current - 1", ylabel="drift_stat")
        ax[2].set(title=f"[{grid}] Detection rate (window = 5 frames)", ylabel="% windows flagged")
        for a in ax:
            a.set_xlabel("drift: rotation in deg | translation in x10 cm")
            a.grid(alpha=.3)
            a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("->", path)


# ---------------------------------------------------------------- failure cases

def edge_debug_view(fr: dict, calib, ring, title: str) -> np.ndarray:
    """Ảnh Canny (xám) + LiDAR depth-edge đã chiếu (màu theo khoảng cách tới image edge: xanh gần, đỏ xa)."""
    dist_map = image_edge_distance(fr["image"])
    exyz, ew = lidar_edge_points(fr["points"], ring)
    uv, _, _ = project_velo_to_image(exyz, calib, fr["image"].shape)
    vis = cv2.cvtColor((dist_map == 0).astype(np.uint8) * 140, cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(fr["image"], 0.45, vis, 0.55, 0)
    d = dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    for (u, v), di in zip(uv.astype(int), d):
        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0) if di <= 3 else (0, 0, 255), -1)
    s = edge_score(exyz, ew, calib, dist_map)
    return put_label(vis, f"{title} | edge score={s:.3f}", 1.4)


def cmd_fail(args) -> None:
    out = Path(args.out_dir) / "figures"
    # 01 Geometry: cùng lệnh "pitch 3°" nhưng trục LiDAR nuScenes (x phải, y trước) khác KITTI (x trước, y trái)
    tiles = []
    for ds, fid in (("kitti", "000011"), ("nuscenes", "scene-0103_010")):
        fr = load_frame(DATASETS[ds], fid)
        row = [render(fr, apply_drift(fr["calib"], k, 3.0), f"{ds}: '{k}' +3 deg", ) for k in ("pitch", "roll")]
        row = [cv2.resize(t, (800, int(800 * t.shape[0] / t.shape[1]))) for t in row]
        tiles.append(np.hstack(row))
    w = min(t.shape[1] for t in tiles)
    p = out / "fail_01_nuscenes_pitch_roll_axis_swap.png"
    cv2.imwrite(str(p), np.vstack([t[:, :w] for t in tiles]))
    print("->", p)

    # 02 Metric/sensor: edge score không phát hiện yaw 2° ở cảnh đêm sau mưa, nhưng phát hiện được ban ngày
    rows = []
    for fid in ("scene-0103_010", "scene-1094_010"):
        fr = load_frame(DATASETS["nuscenes"], fid)
        ring = ring_of("nuscenes", fid, len(fr["points"]))
        pair = [edge_debug_view(fr, apply_drift(fr["calib"], "yaw", lv), ring, f"{fid} yaw {lv:+.0f} deg")
                for lv in (0.0, 2.0)]
        rows.append(np.hstack([cv2.resize(t, (960, 540)) for t in pair]))
    p = out / "fail_02_night_edge_score_misses_yaw2deg.png"
    cv2.imwrite(str(p), np.vstack(rows))
    print("->", p)

    # 03 Metric: % điểm trong FOV gần như không đổi khi yaw lệch -> không dùng được làm cảnh báo drift
    fr = load_frame(DATASETS["kitti"], "000004")
    pts = finite_points(fr["points"])
    objs = object_point_sets(pts, fr["calib"], fr["labels"], fr["image"].shape)
    tiles = []
    for lv in (0.0, 1.0, 2.0):
        cal = apply_drift(fr["calib"], "yaw", lv)
        _, _, m = project_velo_to_image(pts, cal, fr["image"].shape)
        hits = in_box_hits(pts, cal, objs, fr["image"].shape)
        far = [(h, n) for h, n, d in hits if d >= 30]
        pin = 100 * sum(h for h, _ in far) / max(1, sum(n for _, n in far))
        tiles.append(render(fr, cal, f"yaw {lv:+.0f} deg | in-FOV {100 * m.mean():.2f}% | far obj pts in box {pin:.1f}%"))
    p = out / "fail_03_fov_metric_blind_far_objects_000004.png"
    cv2.imwrite(str(p), np.vstack(tiles))
    print("->", p)

    # 04 Time: tắt bù ego-motion giữa thời điểm LiDAR và camera (nuScenes), so % điểm trong box
    rows = []
    for fid in list_frames(DATASETS["nuscenes"]):
        for ego in (True, False):
            fr = load_frame(DATASETS["nuscenes"], fid, use_ego_motion=ego)
            pts = finite_points(fr["points"])
            ref = load_frame(DATASETS["nuscenes"], fid) if not ego else fr  # điểm object xác định bằng calib đúng
            objs = object_point_sets(pts, ref["calib"], ref["labels"], fr["image"].shape)
            hits = in_box_hits(pts, fr["calib"], objs, fr["image"].shape)
            rows.append({"frame": fid, "use_ego_motion": ego,
                         "dt_cam_minus_lidar_ms": (fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000,
                         "hit": sum(h for h, *_ in hits), "tot": sum(n for _, n, _ in hits)})
    df = pd.DataFrame(rows)
    df["scene"] = df.frame.str[:10]
    t = df.groupby(["scene", "use_ego_motion"]).agg(hit=("hit", "sum"), tot=("tot", "sum"),
                                                    mean_abs_dt_ms=("dt_cam_minus_lidar_ms", lambda x: x.abs().mean()))
    t["pct_in_box"] = 100 * t.hit / t.tot
    t = t.reset_index().round(2)
    t.to_csv(Path(args.out_dir) / "time_sync_ego_motion.csv", index=False)
    print(t.to_string(index=False))


# ---------------------------------------------------------------- latency

def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cmd_latency(args) -> None:
    rows = []
    for ds in args.datasets:
        fid = list_frames(DATASETS[ds])[0]
        fr = load_frame(DATASETS[ds], fid)
        pts, shape = fr["points"], fr["image"].shape
        dist_map = image_edge_distance(fr["image"])
        ring = ring_of(ds, fid, len(pts))
        exyz, ew = lidar_edge_points(pts, ring)
        jobs = {
            "project_full_frame": lambda: project_velo_to_image(pts, fr["calib"], shape),
            "edge_score_1_calib": lambda: edge_score(exyz, ew, fr["calib"], dist_map),
            "edge_preprocess(canny+lidar_edges)": lambda: (image_edge_distance(fr["image"]), lidar_edge_points(pts, ring)),
        }
        for name, fn in jobs.items():
            fn()  # warm-up, bỏ lần đầu
            ts = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                fn()
                ts.append(1000 * (time.perf_counter() - t0))
            rows.append({"dataset": ds, "frame": fid, "job": name, "n_points": len(pts), "runs": args.runs,
                         "p50_ms": np.percentile(ts, 50), "p95_ms": np.percentile(ts, 95),
                         "cpu": cpu_name()})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(Path(args.out_dir) / "latency.csv", index=False)
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="LiDAR-camera calibration QA (topic A)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("command", choices=["demo", "sweep", "edge", "fail", "latency", "all"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--kinds", nargs="+", default=["yaw", "pitch", "roll", "tx", "ty", "tz"],
                    help="loại drift cần quét")
    ap.add_argument("--window", type=int, default=5, help="số frame gộp cho một lần quyết định drift (edge)")
    ap.add_argument("--runs", type=int, default=30, help="số lần đo latency (sau 1 lần warm-up)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    Path(args.out_dir, "figures").mkdir(parents=True, exist_ok=True)
    cmds = {"demo": cmd_demo, "sweep": cmd_sweep, "edge": cmd_edge, "fail": cmd_fail, "latency": cmd_latency}
    for name in (cmds if args.command == "all" else [args.command]):
        cmds[name](args)


if __name__ == "__main__":
    main()
