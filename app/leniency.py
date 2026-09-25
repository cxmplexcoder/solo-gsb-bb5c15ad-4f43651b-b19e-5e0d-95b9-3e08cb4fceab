"""评委严宽差异诊断（只读纯函数，仅基于原始分）。

在同一「选手 × 项目」的**共评记录**上，计算每对评委的原始分差中位数与共评数；
再以共评数为权重，把评委看作节点、共评关系看作边，对评委关系图**逐连通分量**
做加权最小二乘拟合，并约束**各分量内评委相对偏移量均值为零**：

    minimize  Σ_{(i,j)} w_ij · ((x_i − x_j) − d_ij)²
    s.t.      Σ_{i ∈ 分量} x_i = 0

其中 ``d_ij`` 为该对评委在共评记录上的原始分差（评委 i − 评委 j）中位数，
``w_ij`` 为共评数。偏移量为正表示该评委相对同分量其他评委打分**偏宽**，
为负则**偏严**。没有任何共评边的评委标记为**不可估计**；偏移量只在同一
连通分量内部可比，**不跨分量比较**。

整个计算只使用**原始分**，不接收裁决、不随校准范围变化；导入或合并数据后
由调用方重新计算。模块不执行任何写操作。
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

# 偏移量绝对值不超过该阈值时视为「居中」（与主排名同分容差一致的量级）。
OFFSET_EPS = 1e-9

DIRECTION_LENIENT = "lenient"  # 偏宽：相对同分量评委打分系统性偏高
DIRECTION_STRICT = "strict"    # 偏严：相对同分量评委打分系统性偏低
DIRECTION_NEUTRAL = "neutral"  # 居中：偏移量约为 0
DIRECTION_LABELS = {
    DIRECTION_LENIENT: "偏宽",
    DIRECTION_STRICT: "偏严",
    DIRECTION_NEUTRAL: "居中",
}


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a square linear system by Gauss-Jordan elimination with pivoting."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= 1e-15:
            raise ArithmeticError("拟合矩阵奇异，无法求解。")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _fit_component(
    names: Sequence[str],
    edges: Sequence[tuple[int, int, float, float]],
) -> dict[str, float]:
    """一个连通分量的加权最小二乘拟合（约束分量内偏移量均值为 0）。

    ``names`` 为分量内评委（节点顺序），``edges`` 为 ``(i, j, weight, diff)``
    且 ``diff`` 的方向为 ``x_i − x_j ≈ diff``。返回 ``评委名 -> 偏移量``。

    法方程为 ``L x = b``（``L`` 为加权拉普拉斯矩阵，``b_i = Σ w·d`` 按边方向
    累加）。``L`` 奇异（秩 n−1，零空间为全 1 向量），给 ``L`` 加上
    ``(1/n)·J``（``J`` 为全 1 矩阵）后矩阵正定可解；由于 ``b`` 的分量和为 0，
    所得解恰满足 ``Σx = 0``，即原约束问题的唯一解。
    """
    n = len(names)
    lap = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for i, j, w, d in edges:
        lap[i][i] += w
        lap[j][j] += w
        lap[i][j] -= w
        lap[j][i] -= w
        rhs[i] += w * d
        rhs[j] -= w * d
    bump = 1.0 / n
    matrix = [[lap[r][c] + bump for c in range(n)] for r in range(n)]
    xs = _solve_linear(matrix, rhs)
    return {name: xs[k] for k, name in enumerate(names)}


def _direction(offset: float) -> str:
    if offset > OFFSET_EPS:
        return DIRECTION_LENIENT
    if offset < -OFFSET_EPS:
        return DIRECTION_STRICT
    return DIRECTION_NEUTRAL


def leniency_diagnosis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """评委严宽差异诊断（纯函数，只读，仅使用原始分）。

    ``records`` 为包含 ``contestant`` / ``judge`` / ``item`` / ``score`` 的
    记录序列（同一三元组有重复时以首条为准，与评委排序分歧一致）。
    """
    # (项目, 选手) -> 评委 -> 原始分
    cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    judge_order: list[str] = []
    for rec in records:
        judge = rec["judge"]
        if judge not in judge_order:
            judge_order.append(judge)
        cell[(rec["item"], rec["contestant"])].setdefault(judge, float(rec["score"]))

    pos = {j: k for k, j in enumerate(judge_order)}

    # 每对评委（按出场顺序定向）在共评记录上的原始分差序列。
    pair_diffs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for scores in cell.values():
        idxs = sorted(pos[j] for j in scores)
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                ia, ib = idxs[x], idxs[y]
                pair_diffs[(ia, ib)].append(
                    scores[judge_order[ia]] - scores[judge_order[ib]]
                )

    # 评委关系图的连通分量（仅有共评边的评委参与；孤立评委不可估计）。
    adjacency: dict[int, list[int]] = defaultdict(list)
    for ia, ib in pair_diffs:
        adjacency[ia].append(ib)
        adjacency[ib].append(ia)
    component_of: dict[int, int] = {}
    components: list[list[int]] = []
    for idx in range(len(judge_order)):
        if idx in component_of or idx not in adjacency:
            continue
        comp_id = len(components)
        stack = [idx]
        component_of[idx] = comp_id
        members: list[int] = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in component_of:
                    component_of[nxt] = comp_id
                    stack.append(nxt)
        components.append(sorted(members))

    judge_rows: dict[str, dict[str, Any]] = {
        j: {
            "judge": j,
            "estimable": False,
            "component": None,
            "offset": None,
            "direction": None,
            "coverage": 0,
            "partners": 0,
        }
        for j in judge_order
    }
    pairs_out: list[dict[str, Any]] = []
    components_out: list[dict[str, Any]] = []
    total_weight = 0
    total_weighted_sq = 0.0

    for comp_id, members in enumerate(components):
        names = [judge_order[k] for k in members]
        local = {g: l for l, g in enumerate(members)}
        # 该分量的共评边（按评委出场顺序排列，方向：出场靠前者 − 靠后者）。
        edges: list[tuple[int, int, float, float]] = []
        edge_keys: list[tuple[int, int]] = []
        for key in sorted(pair_diffs):
            ia, ib = key
            if component_of.get(ia) != comp_id:
                continue
            diffs = pair_diffs[key]
            edges.append((local[ia], local[ib], float(len(diffs)), median(diffs)))
            edge_keys.append(key)

        offsets = _fit_component(names, edges)

        coverage = {name: 0 for name in names}
        partners: dict[str, set[str]] = {name: set() for name in names}
        comp_weight = 0
        comp_weighted_sq = 0.0
        for (i, j, w, d), (ia, ib) in zip(edges, edge_keys):
            ja, jb = judge_order[ia], judge_order[ib]
            coverage[ja] += int(w)
            coverage[jb] += int(w)
            partners[ja].add(jb)
            partners[jb].add(ja)
            fitted = offsets[ja] - offsets[jb]
            residual = d - fitted
            comp_weight += int(w)
            comp_weighted_sq += w * residual * residual
            pairs_out.append(
                {
                    "judge_a": ja,
                    "judge_b": jb,
                    "component": comp_id,
                    "common_count": int(w),
                    "observed_diff": d,
                    "fitted_diff": fitted,
                    "residual": residual,
                }
            )
        total_weight += comp_weight
        total_weighted_sq += comp_weighted_sq

        for name in names:
            row = judge_rows[name]
            row["estimable"] = True
            row["component"] = comp_id
            row["offset"] = offsets[name]
            row["direction"] = _direction(offsets[name])
            row["coverage"] = coverage[name]
            row["partners"] = len(partners[name])

        components_out.append(
            {
                "id": comp_id,
                "judges": names,
                "judge_count": len(names),
                "pair_count": len(edge_keys),
                "co_ratings": comp_weight,
                "weighted_rmse": (
                    (comp_weighted_sq / comp_weight) ** 0.5 if comp_weight else 0.0
                ),
            }
        )

    estimable = sum(1 for row in judge_rows.values() if row["estimable"])
    return {
        "direction_labels": dict(DIRECTION_LABELS),
        "judges": [judge_rows[j] for j in judge_order],
        "pairs": pairs_out,
        "components": components_out,
        "summary": {
            "judges": len(judge_order),
            "estimable_judges": estimable,
            "unestimable_judges": len(judge_order) - estimable,
            "components": len(components_out),
            "judge_pairs": len(pairs_out),
            "co_ratings": total_weight,
            "weighted_rmse": (
                (total_weighted_sq / total_weight) ** 0.5 if total_weight else None
            ),
        },
    }
