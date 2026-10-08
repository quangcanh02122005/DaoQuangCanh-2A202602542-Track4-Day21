"""Metric đo độ khớp LiDAR-camera dưới calibration drift (topic A).

Ba nhóm metric, đều không cần model:
  1. pct_fov      : % điểm LiDAR chiếu được vào khung ảnh.
  2. pct_in_box   : với các điểm nằm trong 3D box GT (xác định bằng calib ĐÚNG, tức "điểm thuộc object"),
                    % điểm vẫn rơi vào 2D box label khi chiếu bằng calib BỊ LỆCH. Chia theo khoảng cách.
  3. edge score   : khớp cạnh độ sâu (depth discontinuity của LiDAR) với cạnh ảnh (Canny), theo ý tưởng
                    Levinson & Thrun, "Automatic Online Calibration of Cameras and Lasers", RSS 2013.
                    Không cần label -> dùng được khi xe chạy thật.
"""
from __future__ import annotations

import cv2
import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

DIST_BINS = [(0.0, 15.0, "near"), (15.0, 30.0, "mid"), (30.0, np.inf, "far")]

# Mỗi loại drift: (tên tham số của perturb_extrinsic, đơn vị hiển thị)
DRIFT_TYPES = {
    "yaw": "deg", "pitch": "deg", "roll": "deg",
    "tx": "m", "ty": "m", "tz": "m",
}


def apply_drift(calib: KittiCalib, kind: str, level: float) -> KittiCalib:
    if kind in ("yaw", "pitch", "roll"):
        return perturb_extrinsic(calib, **{f"{kind}_deg": level})
    t = [0.0, 0.0, 0.0]
    t["xyz".index(kind[1])] = level
    return perturb_extrinsic(calib, t_xyz_m=tuple(t))


def finite_points(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points[:, :3]).all(axis=1)]


# ---------------------------------------------------------------- object points

def points_in_box3d(points_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Mask (N,) điểm (rectified camera frame) nằm trong 3D box KITTI.
    Cùng quy ước với starter.projection.box3d_corners_cam: l theo x, w theo z, y từ -h tới 0."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ R          # = R^T (p - loc) cho từng hàng
    return ((np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 2]) <= w / 2)
            & (local[:, 1] <= 0) & (local[:, 1] >= -h))


def object_point_sets(points: np.ndarray, calib: KittiCalib, labels: list[KittiObject],
                      image_shape, min_points: int = 10) -> list[dict]:
    """Với mỗi object: chỉ số các điểm thuộc object (theo calib đúng) và chiếu được vào ảnh."""
    pc = velo_to_cam(points[:, :3], calib)
    _, _, in_img = cam_to_image(pc, calib.P2, image_shape)
    out = []
    for obj in labels:
        idx = np.flatnonzero(points_in_box3d(pc, obj) & in_img)
        if len(idx) >= min_points:
            out.append({"obj": obj, "idx": idx, "dist": float(np.linalg.norm(obj.location[[0, 2]]))})
    return out


def in_box_hits(points: np.ndarray, calib: KittiCalib, objs: list[dict], image_shape) -> list[tuple[int, int, float]]:
    """(số điểm rơi vào 2D box, tổng số điểm object, khoảng cách) cho từng object, với calib (có thể lệch)."""
    res = []
    for o in objs:
        p = points[o["idx"], :3]
        uv, _, m = cam_to_image(velo_to_cam(p, calib), calib.P2, image_shape)
        x1, y1, x2, y2 = o["obj"].bbox
        hit = int(((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)).sum())
        res.append((hit, len(p), o["dist"]))
    return res


# ---------------------------------------------------------------- edge alignment score

def lidar_edge_points(points: np.ndarray, ring: np.ndarray | None = None,
                      min_jump_m: float = 0.5, max_daz_deg: float = 1.0):
    """Điểm nằm ở mép gần của một bước nhảy độ sâu dọc theo từng scan line.
    KITTI .bin đã theo thứ tự quét từng beam. nuScenes lưu XEN KẼ beam (ring 0,1,..,31,0,1,..),
    nên phải truyền `ring` để sắp xếp lại theo (ring, azimuth), nếu không thì "điểm kề" là beam khác.
    Trả về (xyz (M,3), weight (M,)). weight = sqrt(độ nhảy) như Levinson & Thrun."""
    ok = np.isfinite(points[:, :3]).all(axis=1)
    p = points[ok]
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    if ring is not None:
        order = np.lexsort((az, ring[ok]))
        p, az = p[order], az[order]
    r = np.linalg.norm(p[:, :3], axis=1)
    jump = np.zeros(len(p))
    for k in (-1, 1):  # so với điểm trước và sau trong cùng scan line
        rn, azn = np.roll(r, k), np.roll(az, k)
        same_line = np.abs((az - azn + 180) % 360 - 180) < max_daz_deg
        jump = np.maximum(jump, np.where(same_line, rn - r, 0.0))
    keep = jump > min_jump_m
    return p[keep, :3], np.sqrt(jump[keep])


def image_edge_distance(image: np.ndarray, canny_lo: int = 50, canny_hi: int = 150) -> np.ndarray:
    """Khoảng cách (pixel) từ mỗi pixel tới cạnh Canny gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def edge_score(edge_xyz: np.ndarray, edge_w: np.ndarray, calib: KittiCalib, dist_map: np.ndarray,
               sigma_px: float = 3.0) -> float:
    """Trung bình có trọng số của exp(-d/sigma), d = khoảng cách từ LiDAR edge tới image edge gần nhất.
    1.0 = mọi depth edge nằm đúng trên image edge."""
    uv, _, m = cam_to_image(velo_to_cam(edge_xyz, calib), calib.P2, dist_map.shape)
    if len(uv) == 0:
        return 0.0
    d = dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    w = edge_w[m]
    return float((w * np.exp(-d / sigma_px)).sum() / w.sum())


# Lưới tìm kiếm quanh calib hiện tại: nếu có calib "bên cạnh" khớp cạnh tốt hơn hẳn -> calib hiện tại đã drift.
NEIGHBOURS = ([(k, s * a) for k in ("yaw", "pitch", "roll") for a in (0.5, 1.0, 2.0, 3.0) for s in (-1, 1)]
              + [(k, s * a) for k in ("tx", "ty", "tz") for a in (0.05, 0.10) for s in (-1, 1)])
