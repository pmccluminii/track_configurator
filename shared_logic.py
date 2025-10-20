# shared_logic.py
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

DEFAULT_STOCK = [2.0, 1.0]
TOL = 0.005
MIN_SEGMENT_HARD_M = 0.18
MIN_SEGMENT_WARN_M = 0.36

@dataclass
class Segment:
    length_m: float
    kind: str  # 'stock' or 'cut'
    cut_from: Optional[float] = None

@dataclass
class MidComponent:
    pos_m: float
    part_no: str

@dataclass
class LayoutSpec:
    name: str
    shape: str  # Straight, L, Rectangle, U
    length_m: float
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    stock: List[float] = field(default_factory=lambda: DEFAULT_STOCK.copy())
    max_run_m: Optional[float] = None
    start_end: str = "End cap"
    end_end: str = "End cap"
    corner1_join: str = "Plain join"
    corner2_join: str = "Plain join"
    corner3_join: str = "Plain join"
    mid_components: List[MidComponent] = field(default_factory=list)

def pack_segments(target_len_m: float, stock: List[float], max_run_m: Optional[float]) -> List[Segment]:
    stock = [float(s) for s in stock]
    stock_desc = sorted(stock, reverse=True)
    if not stock_desc:
        return []
    target = round(target_len_m, 3)

    def score_plan(plan: List[Segment]):
        short_pen = sum(1 for seg in plan if seg.length_m < MIN_SEGMENT_WARN_M - 1e-9)
        cut_pen = sum(1 for seg in plan if seg.kind == 'cut')
        seg_count = len(plan)
        waste = sum((seg.cut_from - seg.length_m) for seg in plan if seg.kind == 'cut' and seg.cut_from)
        return (short_pen, cut_pen, seg_count, waste)

    from functools import lru_cache

    @lru_cache(None)
    def best_plan(rem: float, used: float):
        rem = round(rem, 3)
        if rem <= TOL:
            return []
        if rem < MIN_SEGMENT_HARD_M - TOL:
            return None
        best = None
        best_score = None

        for s in stock_desc:
            # final cut option
            if rem <= s + TOL:
                if max_run_m and used + rem > max_run_m + TOL:
                    pass
                else:
                    waste = max(0.0, s - rem)
                    seg_kind = 'stock' if waste <= TOL else 'cut'
                    seg = Segment(rem, seg_kind, cut_from=None if seg_kind == 'stock' else s)
                    plan = [seg]
                    score = score_plan(plan)
                    if best is None or score < best_score:
                        best, best_score = plan, score
                    if abs(rem - s) <= TOL:
                        break
            # stock piece option
            if rem >= s - TOL and rem - s >= -TOL:
                new_used = used + s
                if max_run_m and new_used > max_run_m + TOL:
                    continue
                new_rem = max(0.0, round(rem - s, 3))
                sub = best_plan(new_rem, new_used)
                if sub is None:
                    continue
                plan = [Segment(s, 'stock')] + sub
                score = score_plan(plan)
                if best is None or score < best_score:
                    best, best_score = plan, score
        return best

    plan = best_plan(target, 0.0)
    if plan is None:
        return []
    for seg in plan:
        if seg.kind == 'cut' and seg.cut_from is None:
            donor = next((s for s in stock_desc if s + TOL >= seg.length_m), seg.length_m)
            seg.cut_from = donor
    return plan

def accumulate_bom_for_run(segs: List[Segment]) -> Dict[str, int]:
    bom: Dict[str, int] = {}
    for seg in segs:
        if seg.kind == 'stock':
            key = f"{seg.length_m:.2f} m stock"
        else:
            key = f"Cut from {seg.cut_from:.2f} m -> {seg.length_m:.2f} m"
        bom[key] = bom.get(key, 0) + 1
    return bom

def path_points_for_shape(shape: str, L: float, W: Optional[float], D: Optional[float]):
    if shape == "Straight":
        return [(0,0), (L,0)]
    elif shape == "L":
        if W is None: raise ValueError("Width required for L")
        return [(0,0), (L,0), (L,W)]
    elif shape == "Rectangle":
        if W is None: raise ValueError("Width required for Rectangle")
        return [(0,0), (L,0), (L,W), (0,W), (0,0)]
    elif shape == "U":
        if D is None: raise ValueError("Depth required for U")
        return [(0,0), (0,D), (L,D), (L,0)]
    else:
        raise ValueError("Unsupported shape")

def path_lengths(points):
    lens = []
    tot = 0.0
    for i in range(len(points)-1):
        x1,y1 = points[i]; x2,y2 = points[i+1]
        L = math.hypot(x2-x1, y2-y1)
        lens.append(L); tot += L
    return lens, tot

def place_mid_components(components: List[MidComponent], path_pts):
    seg_lens, tot = path_lengths(path_pts)
    if not seg_lens:
        return []
    cum = [0.0]
    for L in seg_lens: cum.append(cum[-1] + L)
    placed = []
    last_idx = len(seg_lens) - 1
    for mc in components:
        s = max(0.0, min(mc.pos_m, cum[-1]))
        idx = 0
        while idx < last_idx and s > cum[idx+1] - 1e-9:
            idx += 1
        if idx > last_idx:
            idx = last_idx
        seg_len = seg_lens[idx]
        seg_start = cum[idx]
        t = 0 if seg_len == 0 else (s - seg_start) / seg_len
        x1,y1 = path_pts[idx]; x2,y2 = path_pts[idx+1]
        x = x1 + t*(x2-x1); y = y1 + t*(y2-y1)
        placed.append((x,y,mc))
    return placed
