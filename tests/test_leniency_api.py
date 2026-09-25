"""API tests for the read-only judge leniency/severity diagnosis endpoint."""
import os
import sys
import tempfile
import unittest

_tmp = tempfile.mkdtemp(prefix="calib_test_leniency_")
os.environ["CALIB_DATA_DIR"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import database as db  # noqa: E402
from app.main import app  # noqa: E402

db.init_db()
client = TestClient(app)
SID = "leniency-test-session"

CSV_BODY = (
    "选手,评委,项目,分数\n"
    "C1,JA,P1,8\nC1,JB,P1,6\n"
    "C2,JA,P1,7\nC2,JB,P1,5\n"
    "C3,JB,P1,9\nC3,JC,P1,8\n"
    "C4,JA,P1,10\nC4,JC,P1,6\n"
)

# 与 CSV_BODY 相同，仅 C4 的 JA 分数 10 → 7（JA−JC 共评分差随之 4 → 1）。
CSV_MERGED = CSV_BODY.replace("C4,JA,P1,10", "C4,JA,P1,7")


def upload(body=CSV_BODY):
    r = client.post(
        f"/api/sessions/{SID}/upload?scope=judge",
        files={"file": ("s.csv", body.encode("utf-8-sig"), "text/csv")},
    )
    assert r.status_code == 200, r.text
    return r.json()


def get_leniency():
    r = client.get(f"/api/sessions/{SID}/leniency")
    assert r.status_code == 200, r.text
    return r.json()


def judge_row(data, name):
    return next(row for row in data["judges"] if row["judge"] == name)


def pair_row(data, a, b):
    return next(
        p for p in data["pairs"] if p["judge_a"] == a and p["judge_b"] == b
    )


class TestLeniencyAPI(unittest.TestCase):
    def test_01_missing_session_404(self):
        r = client.get("/api/sessions/leniency-no-such-session/leniency")
        self.assertEqual(r.status_code, 404)
        self.assertIn("会话不存在", r.json()["detail"])

    def test_02_session_without_scores_404(self):
        empty_sid = "leniency-empty-session"
        r = client.post(f"/api/sessions/{empty_sid}/scope", json={"scope": "judge"})
        self.assertEqual(r.status_code, 200)  # 会话被创建但没有评分
        r = client.get(f"/api/sessions/{empty_sid}/leniency")
        self.assertEqual(r.status_code, 404)
        self.assertIn("没有评分数据", r.json()["detail"])

    def test_03_upload_and_diagnose(self):
        upload()
        d = get_leniency()
        s = d["summary"]
        self.assertEqual(s["judges"], 3)
        self.assertEqual(s["estimable_judges"], 3)
        self.assertEqual(s["unestimable_judges"], 0)
        self.assertEqual(s["components"], 1)
        self.assertEqual(s["judge_pairs"], 3)
        self.assertEqual(s["co_ratings"], 4)
        self.assertAlmostEqual(s["weighted_rmse"], 0.1 ** 0.5)

        ja, jb, jc = (judge_row(d, n) for n in ("JA", "JB", "JC"))
        self.assertAlmostEqual(ja["offset"], 29 / 15)
        self.assertAlmostEqual(jb["offset"], -4 / 15)
        self.assertAlmostEqual(jc["offset"], -25 / 15)
        self.assertAlmostEqual(ja["offset"] + jb["offset"] + jc["offset"], 0.0)
        self.assertEqual(ja["direction"], "lenient")
        self.assertEqual(jb["direction"], "strict")
        self.assertEqual(ja["coverage"], 3)
        self.assertEqual(jc["coverage"], 2)
        self.assertEqual(d["direction_labels"]["lenient"], "偏宽")

        p_ab = pair_row(d, "JA", "JB")
        self.assertEqual(p_ab["common_count"], 2)
        self.assertAlmostEqual(p_ab["observed_diff"], 2.0)
        self.assertAlmostEqual(p_ab["fitted_diff"], 2.2)
        self.assertAlmostEqual(p_ab["residual"], -0.2)
        p_ac = pair_row(d, "JA", "JC")
        self.assertAlmostEqual(p_ac["observed_diff"], 4.0)
        self.assertAlmostEqual(p_ac["fitted_diff"], 3.6)
        self.assertAlmostEqual(p_ac["residual"], 0.4)

        comp = d["components"][0]
        self.assertEqual(comp["judges"], ["JA", "JB", "JC"])
        self.assertEqual(comp["pair_count"], 3)
        self.assertEqual(comp["co_ratings"], 4)

    def test_04_decisions_do_not_change_result(self):
        upload()
        before = get_leniency()
        analysis = client.get(f"/api/sessions/{SID}/analysis").json()
        score_id = analysis["contestants"][0]["items"][0]["scores"][0]["id"]
        r = client.patch(
            f"/api/sessions/{SID}/scores/{score_id}/decision",
            json={"use_raw": True, "reason": "复核"},
        )
        self.assertEqual(r.status_code, 200)
        after = get_leniency()
        self.assertEqual(before, after)

    def test_05_scope_does_not_change_result(self):
        upload()
        before = get_leniency()
        r = client.post(f"/api/sessions/{SID}/scope", json={"scope": "judge_item"})
        self.assertEqual(r.status_code, 200)
        after = get_leniency()
        self.assertEqual(before, after)
        client.post(f"/api/sessions/{SID}/scope", json={"scope": "judge"})

    def test_06_merge_confirm_refreshes_result(self):
        upload()
        before = get_leniency()
        self.assertAlmostEqual(pair_row(before, "JA", "JC")["observed_diff"], 4.0)

        r = client.post(
            f"/api/sessions/{SID}/merge/preview",
            files={"file": ("m.csv", CSV_MERGED.encode("utf-8-sig"), "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        preview = r.json()
        self.assertEqual(preview["counts"]["changed"], 1)
        r = client.post(
            f"/api/sessions/{SID}/merge/confirm",
            json={"records": preview["records"], "basis": preview["basis"]},
        )
        self.assertEqual(r.status_code, 200, r.text)

        after = get_leniency()
        # 合并后 JA−JC 共评分差中位数 4 → 1，偏移量按新数据重新拟合。
        self.assertAlmostEqual(pair_row(after, "JA", "JC")["observed_diff"], 1.0)
        ja, jb, jc = (judge_row(after, n) for n in ("JA", "JB", "JC"))
        self.assertAlmostEqual(ja["offset"], 17 / 15)
        self.assertAlmostEqual(jb["offset"], -7 / 15)
        self.assertAlmostEqual(jc["offset"], -2 / 3)
        self.assertAlmostEqual(ja["offset"] + jb["offset"] + jc["offset"], 0.0)
        self.assertNotEqual(before, after)

    def test_07_read_only_does_not_touch_session(self):
        upload()
        before = client.get(f"/api/sessions/{SID}/analysis").json()
        get_leniency()
        get_leniency()
        after = client.get(f"/api/sessions/{SID}/analysis").json()
        self.assertEqual(before, after)
        self.assertEqual(client.get(f"/api/sessions/{SID}/rules").json()["rules"], [])


if __name__ == "__main__":
    unittest.main()
