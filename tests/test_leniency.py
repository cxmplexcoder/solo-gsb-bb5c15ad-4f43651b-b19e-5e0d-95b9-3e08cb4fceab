"""Tests for the judge leniency/severity diagnosis (raw-score only, read-only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.leniency import leniency_diagnosis


def recs(data):
    # data: list of (contestant, judge, item, score)
    return [
        {"id": i + 1, "contestant": c, "judge": j, "item": it, "score": s}
        for i, (c, j, it, s) in enumerate(data)
    ]


def judge_row(result, name):
    return next(r for r in result["judges"] if r["judge"] == name)


def pair_row(result, a, b):
    return next(
        p for p in result["pairs"] if p["judge_a"] == a and p["judge_b"] == b
    )


class TestLeniencyDiagnosis(unittest.TestCase):
    def test_two_judges_exact_fit(self):
        # JA−JB 共评分差为 [2, 2, 4]，中位数 2，共评数 3。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 6),
            ("C2", "JA", "P1", 7), ("C2", "JB", "P1", 5),
            ("C3", "JA", "P1", 9), ("C3", "JB", "P1", 5),
        ]
        d = leniency_diagnosis(recs(data))
        s = d["summary"]
        self.assertEqual(s["judges"], 2)
        self.assertEqual(s["estimable_judges"], 2)
        self.assertEqual(s["unestimable_judges"], 0)
        self.assertEqual(s["components"], 1)
        self.assertEqual(s["judge_pairs"], 1)
        self.assertEqual(s["co_ratings"], 3)
        self.assertAlmostEqual(s["weighted_rmse"], 0.0)

        ja, jb = judge_row(d, "JA"), judge_row(d, "JB")
        self.assertTrue(ja["estimable"])
        self.assertTrue(jb["estimable"])
        # 分量内均值 0：偏移量关于 0 对称。
        self.assertAlmostEqual(ja["offset"], 1.0)
        self.assertAlmostEqual(jb["offset"], -1.0)
        self.assertAlmostEqual(ja["offset"] + jb["offset"], 0.0)
        self.assertEqual(ja["direction"], "lenient")
        self.assertEqual(jb["direction"], "strict")
        self.assertEqual(ja["component"], 0)
        self.assertEqual(jb["component"], 0)
        self.assertEqual(ja["coverage"], 3)
        self.assertEqual(jb["coverage"], 3)
        self.assertEqual(ja["partners"], 1)

        p = pair_row(d, "JA", "JB")
        self.assertEqual(p["common_count"], 3)
        self.assertAlmostEqual(p["observed_diff"], 2.0)
        self.assertAlmostEqual(p["fitted_diff"], 2.0)
        self.assertAlmostEqual(p["residual"], 0.0)
        self.assertEqual(p["component"], 0)

        comp = d["components"][0]
        self.assertEqual(comp["id"], 0)
        self.assertEqual(comp["judges"], ["JA", "JB"])
        self.assertEqual(comp["judge_count"], 2)
        self.assertEqual(comp["pair_count"], 1)
        self.assertEqual(comp["co_ratings"], 3)
        self.assertAlmostEqual(comp["weighted_rmse"], 0.0)

    def test_chain_of_three_exact_fit(self):
        # 树状关系图：JA−JB 差 2、JB−JC 差 1（各 1 条共评），拟合无残差。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 6),
            ("C2", "JB", "P1", 9), ("C2", "JC", "P1", 8),
        ]
        d = leniency_diagnosis(recs(data))
        ja, jb, jc = (judge_row(d, n) for n in ("JA", "JB", "JC"))
        self.assertAlmostEqual(ja["offset"], 5 / 3)
        self.assertAlmostEqual(jb["offset"], -1 / 3)
        self.assertAlmostEqual(jc["offset"], -4 / 3)
        self.assertAlmostEqual(ja["offset"] + jb["offset"] + jc["offset"], 0.0)
        self.assertEqual(jb["coverage"], 2)
        self.assertEqual(jb["partners"], 2)
        self.assertEqual(ja["coverage"], 1)
        self.assertEqual(jc["coverage"], 1)
        p_ab = pair_row(d, "JA", "JB")
        p_bc = pair_row(d, "JB", "JC")
        self.assertAlmostEqual(p_ab["fitted_diff"], 2.0)
        self.assertAlmostEqual(p_bc["fitted_diff"], 1.0)
        self.assertAlmostEqual(p_ab["residual"], 0.0)
        self.assertAlmostEqual(p_bc["residual"], 0.0)
        self.assertEqual(d["summary"]["judge_pairs"], 2)
        self.assertEqual(d["summary"]["co_ratings"], 2)

    def test_weighted_cycle_least_squares(self):
        # 环状关系图且观测不一致：JA−JB 差 2（共评 2 条）、JB−JC 差 1、JA−JC 差 4。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 6),
            ("C2", "JA", "P1", 7), ("C2", "JB", "P1", 5),
            ("C3", "JB", "P1", 9), ("C3", "JC", "P1", 8),
            ("C4", "JA", "P1", 10), ("C4", "JC", "P1", 6),
        ]
        d = leniency_diagnosis(recs(data))
        ja, jb, jc = (judge_row(d, n) for n in ("JA", "JB", "JC"))
        # 以共评数为权重的最小二乘解（分量内均值 0）。
        self.assertAlmostEqual(ja["offset"], 29 / 15)
        self.assertAlmostEqual(jb["offset"], -4 / 15)
        self.assertAlmostEqual(jc["offset"], -25 / 15)
        self.assertAlmostEqual(ja["offset"] + jb["offset"] + jc["offset"], 0.0)
        self.assertEqual(ja["direction"], "lenient")
        self.assertEqual(jb["direction"], "strict")
        self.assertEqual(jc["direction"], "strict")
        self.assertEqual(ja["coverage"], 3)
        self.assertEqual(jb["coverage"], 3)
        self.assertEqual(jc["coverage"], 2)

        p_ab = pair_row(d, "JA", "JB")
        self.assertEqual(p_ab["common_count"], 2)
        self.assertAlmostEqual(p_ab["observed_diff"], 2.0)
        self.assertAlmostEqual(p_ab["fitted_diff"], 11 / 5)
        self.assertAlmostEqual(p_ab["residual"], -1 / 5)
        p_bc = pair_row(d, "JB", "JC")
        self.assertEqual(p_bc["common_count"], 1)
        self.assertAlmostEqual(p_bc["observed_diff"], 1.0)
        self.assertAlmostEqual(p_bc["fitted_diff"], 7 / 5)
        self.assertAlmostEqual(p_bc["residual"], -2 / 5)
        p_ac = pair_row(d, "JA", "JC")
        self.assertEqual(p_ac["common_count"], 1)
        self.assertAlmostEqual(p_ac["observed_diff"], 4.0)
        self.assertAlmostEqual(p_ac["fitted_diff"], 18 / 5)
        self.assertAlmostEqual(p_ac["residual"], 2 / 5)

        # 加权残差均方根：sqrt((2·0.2² + 0.4² + 0.4²) / 4) = sqrt(0.1)。
        self.assertAlmostEqual(d["summary"]["weighted_rmse"], 0.1 ** 0.5)
        self.assertAlmostEqual(d["components"][0]["weighted_rmse"], 0.1 ** 0.5)
        self.assertEqual(d["summary"]["co_ratings"], 4)

    def test_median_with_even_common_count(self):
        # 偶数条共评记录时取中间两值均值；两个互不连通的分量各自均值为 0。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 6),   # JA−JB: 2
            ("C2", "JA", "P1", 7), ("C2", "JB", "P1", 6),   # JA−JB: 1
            ("C1", "JC", "P2", 9), ("C1", "JD", "P2", 5),   # JC−JD: 4
            ("C2", "JC", "P2", 8), ("C2", "JD", "P2", 6),   # JC−JD: 2
        ]
        d = leniency_diagnosis(recs(data))
        s = d["summary"]
        self.assertEqual(s["components"], 2)
        self.assertEqual(s["estimable_judges"], 4)
        self.assertEqual(s["judge_pairs"], 2)
        self.assertEqual(s["co_ratings"], 4)

        p_ab = pair_row(d, "JA", "JB")
        self.assertEqual(p_ab["common_count"], 2)
        self.assertAlmostEqual(p_ab["observed_diff"], 1.5)  # median([2, 1])
        self.assertAlmostEqual(p_ab["fitted_diff"], 1.5)
        p_cd = pair_row(d, "JC", "JD")
        self.assertAlmostEqual(p_cd["observed_diff"], 3.0)  # median([4, 2])
        self.assertAlmostEqual(p_cd["fitted_diff"], 3.0)

        ja, jb = judge_row(d, "JA"), judge_row(d, "JB")
        jc, jd = judge_row(d, "JC"), judge_row(d, "JD")
        self.assertAlmostEqual(ja["offset"], 0.75)
        self.assertAlmostEqual(jb["offset"], -0.75)
        self.assertAlmostEqual(jc["offset"], 1.5)
        self.assertAlmostEqual(jd["offset"], -1.5)
        # 各连通分量独立归零，不跨分量比较。
        self.assertEqual(ja["component"], 0)
        self.assertEqual(jb["component"], 0)
        self.assertEqual(jc["component"], 1)
        self.assertEqual(jd["component"], 1)
        self.assertEqual([c["judges"] for c in d["components"]], [["JA", "JB"], ["JC", "JD"]])

    def test_judge_without_common_edge_is_unestimable(self):
        # JC 只评过别人没评过的「选手 × 项目」，没有任何共评边。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 6),
            ("C2", "JA", "P1", 7), ("C2", "JB", "P1", 5),
            ("C3", "JC", "P2", 9),
        ]
        d = leniency_diagnosis(recs(data))
        jc = judge_row(d, "JC")
        self.assertFalse(jc["estimable"])
        self.assertIsNone(jc["component"])
        self.assertIsNone(jc["offset"])
        self.assertIsNone(jc["direction"])
        self.assertEqual(jc["coverage"], 0)
        self.assertEqual(jc["partners"], 0)
        s = d["summary"]
        self.assertEqual(s["judges"], 3)
        self.assertEqual(s["estimable_judges"], 2)
        self.assertEqual(s["unestimable_judges"], 1)
        self.assertEqual(s["components"], 1)
        # 不可估计评委不产生任何评委对记录。
        self.assertEqual(len(d["pairs"]), 1)
        self.assertEqual(d["pairs"][0]["judge_a"], "JA")
        self.assertEqual(d["pairs"][0]["judge_b"], "JB")

    def test_no_common_ratings_at_all(self):
        # 两位评委从未评过同一「选手 × 项目」：全部不可估计，无评委对。
        data = [
            ("C1", "JA", "P1", 8),
            ("C2", "JB", "P1", 7),
        ]
        d = leniency_diagnosis(recs(data))
        self.assertEqual(d["pairs"], [])
        self.assertEqual(d["components"], [])
        self.assertTrue(all(not r["estimable"] for r in d["judges"]))
        s = d["summary"]
        self.assertEqual(s["judges"], 2)
        self.assertEqual(s["estimable_judges"], 0)
        self.assertEqual(s["unestimable_judges"], 2)
        self.assertEqual(s["components"], 0)
        self.assertEqual(s["judge_pairs"], 0)
        self.assertEqual(s["co_ratings"], 0)
        self.assertIsNone(s["weighted_rmse"])

    def test_empty_records(self):
        d = leniency_diagnosis([])
        self.assertEqual(d["judges"], [])
        self.assertEqual(d["pairs"], [])
        self.assertEqual(d["components"], [])
        s = d["summary"]
        self.assertEqual(s["judges"], 0)
        self.assertEqual(s["estimable_judges"], 0)
        self.assertEqual(s["judge_pairs"], 0)
        self.assertEqual(s["co_ratings"], 0)
        self.assertIsNone(s["weighted_rmse"])
        self.assertEqual(
            d["direction_labels"],
            {"lenient": "偏宽", "strict": "偏严", "neutral": "居中"},
        )

    def test_neutral_direction_when_offset_zero(self):
        # JA/JB 共评分差为 0：偏移量为 0，方向居中。
        data = [
            ("C1", "JA", "P1", 8), ("C1", "JB", "P1", 8),
            ("C2", "JA", "P1", 6), ("C2", "JB", "P1", 6),
        ]
        d = leniency_diagnosis(recs(data))
        ja = judge_row(d, "JA")
        self.assertAlmostEqual(ja["offset"], 0.0)
        self.assertEqual(ja["direction"], "neutral")
        p = pair_row(d, "JA", "JB")
        self.assertAlmostEqual(p["observed_diff"], 0.0)
        self.assertAlmostEqual(p["residual"], 0.0)

    def test_co_rating_unit_is_contestant_item_pair(self):
        # 同一选手不同项目不算共评；同一项目不同选手也不算共评。
        data = [
            ("C1", "JA", "P1", 8),
            ("C1", "JB", "P2", 6),   # 同选手不同项目：不算
            ("C2", "JB", "P1", 7),   # 同项目不同选手：不算
        ]
        d = leniency_diagnosis(recs(data))
        self.assertEqual(d["pairs"], [])
        self.assertEqual(d["summary"]["estimable_judges"], 0)
        # 追加一条真正的共评记录后出现评委对。
        data2 = data + [("C1", "JB", "P1", 5)]
        d2 = leniency_diagnosis(recs(data2))
        self.assertEqual(d2["summary"]["judge_pairs"], 1)
        p = pair_row(d2, "JA", "JB")
        self.assertEqual(p["common_count"], 1)
        self.assertAlmostEqual(p["observed_diff"], 3.0)


if __name__ == "__main__":
    unittest.main()
