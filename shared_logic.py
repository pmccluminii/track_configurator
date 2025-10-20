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
    """
    Slice a requested length into stock pieces, preferring:
      1. exact stock lengths (no joins),
      2. a single cut from the smallest stick that covers the remainder,
      3. otherwise the classic greedy selection while respecting min-segment limits.
    """
    remaining = round(target_len_m, 3)
    segs: List[Segment] = []
    run_acc = 0.0
    stock_desc = sorted(stock, reverse=True)
    stock_asc = sorted(stock)

    def push_len(L):
        nonlocal run_acc
        segs.append(Segment(L, 'stock'))
        run_acc += L

    while remaining > TOL:
        placed = False
        exact_candidates = []
        viable = []
        for s in stock_desc:
            if remaining >= s - 1e-6:
                if max_run_m and run_acc + s > max_run_m:
                    continue
                leftover = round(remaining - s, 3)
                if abs(leftover) <= TOL:
                    exact_candidates.append((s, leftover))
                    continue
                # Skip choices that would create a tiny trailing piece; we'll try a smaller stock or cut instead.
                if leftover > TOL and leftover < MIN_SEGMENT_HARD_M - 1e-9:
                    continue
                viable.append((s, leftover))
        if exact_candidates:
            s, _ = exact_candidates[0]
            push_len(s)
            remaining = 0.0
            placed = True
        elif viable:
            high = [item for item in viable if item[1] >= MIN_SEGMENT_WARN_M - 1e-9]
            if high:
                # Prefer smallest leftover ≥ warning threshold (reduces remainder while keeping it workable)
                s, _ = min(high, key=lambda item: (item[1], item[0]))
            else:
                # No safe leftover ≥ warn; choose the option that leaves the largest possible remainder to avoid very short segments later
                s, _ = max(viable, key=lambda item: (item[1], -item[0]))
            push_len(s)
            remaining = round(remaining - s, 3)
            placed = True
        if placed:
            continue

        cut_candidate = next((s for s in stock_asc if s >= remaining - 1e-6), None)
        if cut_candidate is None:
            cut_candidate = stock_desc[-1] if stock_desc else remaining
        segs.append(Segment(remaining, 'cut', cut_from=cut_candidate))
        remaining = 0.0
    return segs

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
