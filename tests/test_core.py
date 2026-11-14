import datetime as dt
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acme_agent import validators as V  # noqa: E402
from acme_agent.agent import Conversation, _misses_field, looks_garbled  # noqa: E402
from harness.fakes import FakeDB, ScriptedLLM  # noqa: E402

TODAY = dt.date(2026, 11, 14)
CLOCK = lambda: dt.datetime(2026, 11, 14, 12, 0, tzinfo=dt.timezone.utc)  # noqa: E731


def conv(cid, script, db=None, identity=False):
    db = db or FakeDB()
    c, _ = Conversation.start(cid, db, ScriptedLLM(script), CLOCK)
    c.s["identity"] = True if identity else None
    return c, db


def save(*items):
    return ("save_fields", {"fields": [{"field": f, "value": v, "evidence": e} for f, v, e in items]})


class Validators(unittest.TestCase):
    def test_enums_and_numbers(self):
        self.assertEqual(V.validate("degree", "btech").value, "B.Tech")
        self.assertEqual(V.validate("preferred_location", "Bangalore").value, "Bengaluru")
        self.assertFalse(V.validate("degree", "BBA").ok)
        self.assertFalse(V.validate("cgpa", 85).ok)
        self.assertFalse(V.validate("cgpa", "84%").ok)
        self.assertEqual(V.validate("cgpa", "8.456").value, 8.46)
        self.assertTrue(V.validate("cgpa", 3.5).confirm)
        self.assertFalse(V.validate("graduation_year", 2025).ok)
        self.assertFalse(V.validate("internship_duration_months", 4).ok)
        self.assertEqual(V.validate("internship_duration_months", "three months").value, 3)
        self.assertFalse(V.validate("available_from", "2028-01-01", TODAY).ok)
        self.assertFalse(V.validate("available_from", "2027-02-30", TODAY).ok)
        self.assertFalse(V.validate("email", "bad@@x").ok)
        self.assertTrue(V.validate("email", "a@gmial.com").confirm)
        self.assertFalse(V.validate("github_url", "github.com/u/repo").ok)
        self.assertEqual(V.validate("linkedin_url", "linkedin.com/in/aa/").value, "https://linkedin.com/in/aa")

    def test_audit(self):
        db = FakeDB()
        a = V.audit_record(db.rows["c003"], today=TODAY)
        self.assertEqual({i["field"] for i in a.invalid}, {"cgpa", "degree"})
        self.assertTrue(V.audit_record(db.rows["c002"], today=TODAY).complete)
        self.assertIn("college", V.audit_record(db.rows["c007"], today=TODAY).missing)
        self.assertEqual(V.audit_record(db.rows["c005"], today=TODAY).autofix, {"degree": "B.Tech", "cgpa": 8.5})


class AgentGuards(unittest.TestCase):
    def test_complete_record_starts_no_chat(self):
        c, db = conv("c002", [])
        self.assertTrue(c.s["ended"])
        self.assertEqual(c.s["outcome"], "no_action")
        self.assertEqual(db.rows["c002"]["collection_status"], "completed")
        self.assertEqual(db.rows["c002"]["attempt_count"], 0)

    def test_cannot_save_before_identity(self):
        c, db = conv("c001", [save(("branch", "CS", "CS")), "hi"])
        c.turn("CS")
        self.assertEqual(db.writes, [])

    def test_hallucinated_value_rejected(self):
        c, db = conv("c001", [save(("cgpa", 9.1, "my cgpa is 8.4")), "ok"], identity=True)
        c.turn("my cgpa is 8.4")
        self.assertIsNone(db.rows["c001"]["cgpa"])

    def test_fabricated_evidence_rejected(self):
        c, db = conv("c001", [save(("email", "z@z.com", "z@z.com")), "ok"], identity=True)
        c.turn("I would rather not say")
        self.assertEqual(db.writes, [])

    def test_good_save_and_no_overwrite(self):
        c, db = conv("c001", [save(("branch", "computer science", "I study computer science"),
                                   ("college", "IIT Delhi", "IIT Delhi")), "done"], identity=True)
        c.turn("I study computer science at IIT Delhi")
        r = db.rows["c001"]
        self.assertEqual(r["branch"], "Computer Science")
        self.assertEqual(r["college"], "VJTI Mumbai")
        self.assertEqual(r["collection_status"], "in_progress")
        self.assertEqual(len(db.writes), 1)

    def test_claim_without_save_is_blocked(self):
        c, db = conv("c001", ["I've saved your details!", "Which college do you attend?"], identity=True)
        self.assertIn("college", c.turn("my cgpa is 8.4"))

    def test_leak_before_identity_blocked(self):
        c, db = conv("c001", ["Is aarav@example.com still correct?", "Am I speaking with Aarav Sharma?"])
        self.assertIn("Aarav", c.turn("hello"))

    def test_completed_rejected_if_incomplete(self):
        c, db = conv("c001", [("end_conversation", {"outcome": "completed", "summary": "all done", "final_message": "bye"}),
                              "still need a few things"], identity=True)
        c.turn("that is all")
        self.assertFalse(c.s["ended"])
        self.assertEqual(db.rows["c001"]["collection_status"], "pending")

    def test_full_flow_completed(self):
        script = [save(("branch", "Electronics", "Electronics"), ("graduation_year", 2027, "2027"),
                       ("cgpa", 8.2, "8.2"), ("preferred_location", "Remote", "remote"),
                       ("available_from", "2027-01-15", "15 jan 2027"), ("internship_duration_months", 6, "6 months")),
                  ("end_conversation", {"outcome": "completed", "summary": "All collected", "final_message": "Thanks!"})]
        c, db = conv("c001", script, identity=True)
        out = c.turn("Electronics, 2027, 8.2, remote, 15 jan 2027, 6 months")
        r = db.rows["c001"]
        self.assertTrue(c.s["ended"])
        self.assertEqual((r["collection_status"], r["attempt_count"]), ("completed", 1))
        self.assertTrue(r["last_contacted_at"].endswith("+05:30"))
        self.assertEqual(out, "Thanks!")

    def test_abandon_paths(self):
        c, db = conv("c001", [])
        c.abandon("no reply")
        r = db.rows["c001"]
        self.assertEqual((r["collection_status"], r["attempt_count"]), ("no_answer", 1))
        c, db = conv("c001", [save(("branch", "IT", "IT")), "ok"], identity=True)
        c.turn("IT")
        c.abandon("left")
        self.assertEqual(db.rows["c001"]["collection_status"], "in_progress")
        self.assertIn("Still needed", db.rows["c001"]["notes"])

    def test_conflict_and_outage(self):
        db = FakeDB(conflict_next=1)
        c, _ = conv("c004", [save(("branch", "IT", "IT")), "ok"], db, identity=True)
        c.turn("IT")
        self.assertEqual(db.rows["c004"]["branch"], "IT")
        db = FakeDB(fail_next=10)
        c, _ = conv("c004", [save(("branch", "IT", "IT")), "ok"], db, identity=True)
        c.turn("IT")
        self.assertEqual(c.s["unsaved"], {"branch": "IT"})

    def test_attempt_cap(self):
        c, db = conv("c008", [])
        c.abandon("x")
        self.assertEqual(db.rows["c008"]["attempt_count"], 50)

    def test_off_topic_answer_is_flagged(self):
        self.assertTrue(_misses_field("email", "haha I like cricket"))
        self.assertFalse(_misses_field("email", "use meera at college dot edu"))
        self.assertTrue(_misses_field("cgpa", "I graduate next year"))
        self.assertFalse(_misses_field("cgpa", "it's 8.6"))

    def test_garbled_reply_never_reaches_history(self):
        junk = "xq" * 80
        self.assertTrue(looks_garbled(junk))
        c, _ = conv("c001", [junk, junk], identity=True)
        spoken = c.turn("lol")
        self.assertNotIn(junk[:12], spoken)
        self.assertNotIn(junk[:12], json.dumps(c.s["messages"]))
        self.assertIn("your application", spoken)

    def test_autofix_piggybacks(self):
        c, db = conv("c005", [])
        self.assertTrue(c.s["ended"])
        r = db.rows["c005"]
        self.assertEqual((r["degree"], r["cgpa"], r["collection_status"]), ("B.Tech", 8.5, "completed"))


if __name__ == "__main__":
    unittest.main()
