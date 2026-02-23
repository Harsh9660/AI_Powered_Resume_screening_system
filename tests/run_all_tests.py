"""
=============================================================
  AI-Powered Resume Screening System — Full Test Suite
=============================================================
Covers:
  ✅ Unit Tests       — individual components in isolation
  ✅ Integration Tests— components wired together
  ✅ Cross Tests      — edge cases and adversarial inputs
  ✅ Automation Test  — live API end-to-end via HTTP curl
=============================================================
Run:  export PYTHONPATH=. && python3 tests/run_all_tests.py
"""

import sys, os, io, time, json, subprocess, textwrap, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF

# ─── colour helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = []
failed = []

def ok(name):
    print(f"  {GREEN}✔ PASS{RESET}  {name}")
    passed.append(name)

def fail(name, reason=""):
    print(f"  {RED}✘ FAIL{RESET}  {name}")
    if reason:
        print(f"         {YELLOW}{reason}{RESET}")
    failed.append(name)

def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ─── PDF helper ───────────────────────────────────────────────────────────────
def make_pdf(lines: list[str], path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in lines:
        pdf.multi_cell(0, 8, txt=line)
    pdf.output(path)
    return path

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 – UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("UNIT TESTS — Parser")

from src.utils.parser import parse_pdf

# U1: Parse a real PDF file
p = make_pdf(["Python Developer", "Skills: Python, Docker, AWS, Machine Learning"], "/tmp/u1_resume.pdf")
text = parse_pdf(p)
if "python" in text.lower() and len(text) > 10:
    ok("U1 — parse_pdf reads real PDF text")
else:
    fail("U1 — parse_pdf reads real PDF text", f"Got: {repr(text[:80])}")

# U2: Empty PDF (blank page) returns empty string gracefully
from fpdf import FPDF as _FPDF
_b = _FPDF()
_b.add_page()
_b.output("/tmp/u2_blank.pdf")
text2 = parse_pdf("/tmp/u2_blank.pdf")
if text2 == "":
    ok("U2 — parse_pdf returns '' for blank PDF")
else:
    fail("U2 — parse_pdf returns '' for blank PDF", f"Got: {repr(text2[:40])}")

# U3: parse_pdf handles bad path without crash
res = parse_pdf("/tmp/nonexistent_12345.pdf")
if res == "":
    ok("U3 — parse_pdf returns '' on bad path (no crash)")
else:
    fail("U3 — parse_pdf bad path", f"Got: {repr(res)}")

# ─────────────────────────────────────────────────────────────────────────────
section("UNIT TESTS — ResumeProcessor")

from src.utils.processor import ResumeProcessor
proc = ResumeProcessor()

# U4: extract_skills on a known sentence
skills = proc.extract_skills("I know Python, Docker and Machine Learning")
skills_lower = {s.lower() for s in skills}
if "python" in skills_lower and "docker" in skills_lower:
    ok("U4 — extract_skills detects Python and Docker")
else:
    fail("U4 — extract_skills detects Python and Docker", f"Got: {skills}")

# U5: abbreviation expansion — ML → Machine Learning
skills5 = proc.extract_skills("I work with ML and AWS daily")
skills5_lower = {s.lower() for s in skills5}
if "machine learning" in skills5_lower:
    ok("U5 — abbreviation expansion: ML → Machine Learning")
else:
    fail("U5 — abbreviation expansion: ML → Machine Learning", f"Got: {skills5}")

# U6: process_text returns cleaned_text
res6 = proc.process_text("Python developer with Docker experience")
if "cleaned_text" in res6 and len(res6["cleaned_text"]) > 5:
    ok("U6 — process_text returns cleaned_text")
else:
    fail("U6 — process_text", f"Got: {res6}")

# U7: extract_skills on empty string returns empty list
skills7 = proc.extract_skills("")
if isinstance(skills7, list) and len(skills7) == 0:
    ok("U7 — extract_skills('') returns [] with no crash")
else:
    fail("U7 — extract_skills empty string", f"Got: {skills7}")

# ─────────────────────────────────────────────────────────────────────────────
section("UNIT TESTS — ScoringEngine")

from src.components.scoring_engine import ScoringEngine
engine = ScoringEngine()

# U8: Strong match scores higher than weak match
strong_resume  = "Python ML Engineer. Skills: Python, Machine Learning, AWS, Docker, TensorFlow, REST API"
weak_resume    = "Marketing Manager with Excel and PowerPoint skills. No coding experience."
jd             = "Looking for a Python ML Engineer with Docker, AWS, TensorFlow and REST API"

jd_skills      = set(proc.extract_skills(jd))
s_skills       = set(proc.extract_skills(strong_resume))
w_skills       = set(proc.extract_skills(weak_resume))

s_result = engine.score(strong_resume, jd, s_skills, jd_skills)
w_result = engine.score(weak_resume,   jd, w_skills, jd_skills)

if s_result["score"] > w_result["score"]:
    ok(f"U8 — strong match ({s_result['score']:.2f}) > weak match ({w_result['score']:.2f})")
else:
    fail("U8 — strong > weak score", f"strong={s_result['score']}, weak={w_result['score']}")

# U9: Score is bounded [0, 1]
for name, r in [("strong", s_result), ("weak", w_result)]:
    if 0.0 <= r["score"] <= 1.0:
        ok(f"U9 — {name} score in [0,1]: {r['score']:.4f}")
    else:
        fail(f"U9 — score out of bounds for {name}", str(r["score"]))

# U10: explain() returns 3 keys
exp = engine.explain(s_result["signals"])
if len(exp) == 3:
    ok("U10 — explain() returns 3 signal contributions")
else:
    fail("U10 — explain()", f"Got {len(exp)} keys: {exp}")

# U11: score() handles empty text without crash
try:
    r11 = engine.score("", "", set(), set())
    ok(f"U11 — score() handles empty inputs, returns {r11['score']}")
except Exception as e:
    fail("U11 — score() empty inputs", str(e))

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 – INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("INTEGRATION TESTS — Parser + Processor + Engine")

# I1: Full pipeline on a PDF file
pdf_path = make_pdf([
    "Jane Smith — Data Scientist",
    "5 years experience in Python, Machine Learning, TensorFlow, Docker, AWS",
    "Worked on NLP projects using PyTorch. Deployed models on GCP.",
    "Led Agile teams. Strong REST API design experience."
], "/tmp/i1_jane.pdf")

text_i1 = parse_pdf(pdf_path)
skills_i1 = set(proc.extract_skills(text_i1))
jd_i1 = "Senior ML Engineer: Python, Machine Learning, Docker, AWS, TensorFlow, REST API, NLP"
jd_skills_i1 = set(proc.extract_skills(jd_i1))
score_i1 = engine.score(text_i1, jd_i1, skills_i1, jd_skills_i1)

if score_i1["score"] >= 0.40:
    ok(f"I1 — Full pipeline: well-matched PDF scores {score_i1['score']*100:.1f}% (≥40%)")
else:
    fail("I1 — Full pipeline score too low", f"score={score_i1['score']}, skills={skills_i1}")

# I2: Weak resume scores lower
pdf_path2 = make_pdf([
    "Tom Brown — Barista",
    "Experience making coffee, managing a café, customer service",
    "Microsoft Word, Excel usage"
], "/tmp/i2_tom.pdf")
text_i2 = parse_pdf(pdf_path2)
skills_i2 = set(proc.extract_skills(text_i2))
score_i2 = engine.score(text_i2, jd_i1, skills_i2, jd_skills_i1)
if score_i2["score"] < score_i1["score"]:
    ok(f"I2 — Weak PDF ({score_i2['score']*100:.1f}%) < strong PDF ({score_i1['score']*100:.1f}%)")
else:
    fail("I2 — Weak < strong", f"weak={score_i2['score']}, strong={score_i1['score']}")

# I3: Multi-resume ranking order
pdf3 = make_pdf(["Mid-level dev: Python, Docker"], "/tmp/i3_mid.pdf")
text_i3 = parse_pdf(pdf3)
skills_i3 = set(proc.extract_skills(text_i3))
score_i3 = engine.score(text_i3, jd_i1, skills_i3, jd_skills_i1)

ranks  = sorted([
    ("Jane (strong)",  score_i1["score"]),
    ("Mid-level",      score_i3["score"]),
    ("Tom (weak)",     score_i2["score"]),
], key=lambda x: x[1], reverse=True)

if ranks[0][0] == "Jane (strong)" and ranks[-1][0] == "Tom (weak)":
    ok(f"I3 — Ranking order correct: {' > '.join(n for n,_ in ranks)}")
else:
    fail("I3 — Ranking order", str(ranks))

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 – CROSS / EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("CROSS TESTS — Edge Cases & Adversarial Inputs")

# C1: Resume that copies the entire JD verbatim should score high
verbatim = "Looking for a Python ML Engineer with Docker, AWS, TensorFlow and REST API"
v_skills = set(proc.extract_skills(verbatim))
v_result = engine.score(verbatim, jd_i1, v_skills, jd_skills_i1)
if v_result["score"] >= 0.50:
    ok(f"C1 — Verbatim JD copy scores high: {v_result['score']*100:.1f}%")
else:
    fail("C1 — Verbatim JD copy score", str(v_result["score"]))

# C2: JD with no known skills — score should not crash
jd_no_skills = "We are a startup seeking passionate people who love challenges"
jd_ns_skills = set(proc.extract_skills(jd_no_skills))
try:
    r_c2 = engine.score(strong_resume, jd_no_skills, s_skills, jd_ns_skills)
    ok(f"C2 — JD with no skills doesn't crash: score={r_c2['score']:.4f}")
except Exception as e:
    fail("C2 — JD no skills", str(e))

# C3: Same resume vs very different JDs — scores should differ
jd_frontend = "Frontend developer: React, JavaScript, CSS, HTML, Vue, Angular"
jd_devops   = "DevOps Engineer: Docker, Kubernetes, CI/CD, Terraform, Jenkins, Linux"
jd_fe_skills = set(proc.extract_skills(jd_frontend))
jd_do_skills = set(proc.extract_skills(jd_devops))

backend_resume = "Sr Python developer: Python, Django, REST API, PostgreSQL"
br_skills = set(proc.extract_skills(backend_resume))

fe_score = engine.score(backend_resume, jd_frontend, br_skills, jd_fe_skills)["score"]
do_score = engine.score(backend_resume, jd_devops,   br_skills, jd_do_skills)["score"]

if fe_score != do_score:
    ok(f"C3 — Same resume different JDs → different scores: FE={fe_score:.3f}, DevOps={do_score:.3f}")
else:
    fail("C3 — Different JDs same score", f"fe={fe_score}, devops={do_score}")

# C4: Unicode / special characters in resume
unicode_resume = "Développeur Python • Compétences: Docker, AWS, Machine Learning"
try:
    u_skills = set(proc.extract_skills(unicode_resume))
    r_c4 = engine.score(unicode_resume, jd_i1, u_skills, jd_skills_i1)
    ok(f"C4 — Unicode resume doesn't crash: score={r_c4['score']:.4f}")
except Exception as e:
    fail("C4 — Unicode resume", str(e))

# C5: Very long resume text (10K words) doesn't hang
long_resume = ("Python Machine Learning Docker AWS TensorFlow REST API " * 300)
try:
    t0 = time.time()
    lr_skills = set(proc.extract_skills(long_resume))
    r_c5 = engine.score(long_resume, jd_i1, lr_skills, jd_skills_i1)
    elapsed = time.time() - t0
    if elapsed < 10:
        ok(f"C5 — Long resume (300× phrase) completes in {elapsed:.1f}s, score={r_c5['score']:.4f}")
    else:
        fail("C5 — Long resume too slow", f"{elapsed:.1f}s")
except Exception as e:
    fail("C5 — Long resume", str(e))

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 – LIVE API AUTOMATION TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("AUTOMATION TESTS — Live API (http://localhost:8000)")

API = "http://localhost:8000"

def api_up():
    try:
        r = subprocess.run(["curl", "-s", f"{API}/"], capture_output=True, text=True, timeout=5)
        return "Welcome" in r.stdout
    except Exception:
        return False

if not api_up():
    print(f"  {YELLOW}⚠ API not running — skipping automation tests.{RESET}")
    print(f"  Start it with: export PYTHONPATH=. && python3 src/api/main.py")
else:
    # A1: GET / health check
    r = subprocess.run(["curl", "-s", f"{API}/"], capture_output=True, text=True)
    if "Welcome" in r.stdout:
        ok("A1 — GET / health check returns 200")
    else:
        fail("A1 — GET / health check", r.stdout[:100])

    # A2: POST /rank-resumes with strong resume
    strong_pdf = make_pdf([
        "Alice Engineer — ML Engineer",
        "6 years Python, TensorFlow, Machine Learning, Docker, AWS, REST API",
        "Led NLP team. Deployed Keras models on AWS. CI/CD with Jenkins."
    ], "/tmp/a2_strong.pdf")

    weak_pdf = make_pdf([
        "Bob — Office Manager",
        "Experience with Excel, PowerPoint, and scheduling meetings."
    ], "/tmp/a2_weak.pdf")

    jd_text = "Hiring Python ML Engineer: Python, Machine Learning, Docker, AWS, TensorFlow, REST API"

    cmd = [
        "curl", "-s", "-X", "POST", f"{API}/rank-resumes",
        "-F", f"job_description={jd_text}",
        "-F", f"resumes=@{strong_pdf}",
        "-F", f"resumes=@{weak_pdf}",
    ]
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        data2 = json.loads(r2.stdout)
        ranked = data2.get("ranked_resumes", [])
        if len(ranked) == 2:
            ok(f"A2 — POST returns 2 ranked resumes")
        else:
            fail("A2 — Expected 2 resumes", f"Got {len(ranked)}: {r2.stdout[:200]}")

        # A3: First ranked resume should be the strong one
        if ranked[0]["score"] > ranked[1]["score"]:
            ok(f"A3 — Stronger resume ranked first ({ranked[0]['score']*100:.1f}% > {ranked[1]['score']*100:.1f}%)")
        else:
            fail("A3 — Wrong ranking order", f"scores: {[r['score'] for r in ranked]}")

        # A4: Scores are non-zero
        if all(r["score"] > 0 for r in ranked):
            ok(f"A4 — All scores non-zero: {[round(r['score']*100,1) for r in ranked]}%")
        else:
            fail("A4 — Zero score detected", str(ranked))

        # A5: Skills field is a list
        for res in ranked:
            if not isinstance(res.get("skills"), list):
                fail(f"A5 — skills field not a list for {res['filename']}")
                break
        else:
            ok("A5 — All resumes have skills list")

        # A6: Explanation contains 3 signal keys
        for res in ranked:
            exp_keys = set(res.get("explanation", {}).keys())
            if len(exp_keys) == 3:
                ok(f"A6 — Explanation has 3 keys for {res['filename']}")
            else:
                fail(f"A6 — Explanation keys for {res['filename']}", str(exp_keys))

    except json.JSONDecodeError:
        fail("A2/A3/A4/A5/A6", f"Invalid JSON from API: {r2.stdout[:300]}")

    # A7: Edge — empty job description
    cmd7 = [
        "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
        "-X", "POST", f"{API}/rank-resumes",
        "-F", "job_description= ",
        "-F", f"resumes=@{strong_pdf}",
    ]
    r7 = subprocess.run(cmd7, capture_output=True, text=True, timeout=30)
    # Should return 200 (we handle empty, not reject)
    if r7.stdout.strip() in ("200", "422"):
        ok(f"A7 — Empty JD returns HTTP {r7.stdout.strip()} (no unhandled crash)")
    else:
        fail("A7 — Empty JD HTTP code", r7.stdout.strip())

# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}{RESET}")
total = len(passed) + len(failed)
print(f"{BOLD}  RESULTS: {GREEN}{len(passed)} passed{RESET}{BOLD}, {RED}{len(failed)} failed{RESET}{BOLD} / {total} total{RESET}")
print(f"{BOLD}{'='*60}{RESET}")

if failed:
    print(f"\n{RED}Failed tests:{RESET}")
    for f in failed:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"\n{GREEN}🎉 All tests passed!{RESET}\n")
    sys.exit(0)
