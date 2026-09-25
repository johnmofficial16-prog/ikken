"""Synthetic states and typed questions with deterministic answers.

Two record types, rendered as text:
  * service-metrics record (categorical fields + daily cloud spend, totals, error rate this/last week)
  * support ticket (customer, plan, product, priority, assignee, SLA hours left, age)
Every question is a Choice question (2-30 options); yes/no questions are 2-option choices.
Each question carries a `family`: "lookup" (copy a categorical field) or "numeric"
(threshold / comparison / argmax / bucketing over numbers in the record).

`numeric_style` controls how numbers are rendered (experiment 4):
  raw     - "$1,234.56", "2.70%"
  digits  - digits separated by spaces: "$ 1 2 3 4 . 5 6"
  derived - raw, plus computed facts appended as text (error-rate change, spend vs budget ...)
"""
from __future__ import annotations

import random

COUNTRIES = ["Germany", "France", "Spain", "Italy", "Poland", "Netherlands", "Belgium", "Sweden", "Norway",
             "Denmark", "Finland", "Ireland", "Portugal", "Austria", "Switzerland", "Greece", "Turkey",
             "Brazil", "Mexico", "Canada", "Argentina", "Chile", "Japan", "India", "Australia",
             "Indonesia", "Vietnam", "Egypt", "Nigeria", "Kenya"]
REGIONS = ["eu-west", "eu-north", "us-east", "us-west", "ap-south", "ap-east", "sa-east", "af-south"]
TIERS = ["web", "api", "batch", "streaming", "storage", "analytics"]
STATUSES = ["healthy", "degraded", "maintenance", "scaling", "retired"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AGENTS = ["Priya", "Marco", "Aisha", "Tomasz", "Chen", "Lucia", "Omar", "Hannah", "Kenji", "Fatima",
          "Diego", "Ingrid", "Ravi", "Sofia", "Kwame", "Elena", "Mateo", "Yuki", "Noah", "Amara",
          "Lars", "Mei", "Jonas", "Leila", "Pedro", "Anika", "Hugo", "Zara", "Felix", "Nadia"]
PLANS = ["Free", "Starter", "Pro", "Business", "Enterprise"]
CHANNELS = ["email", "chat", "phone", "web form", "social"]
PRODUCTS = ["Billing", "Login", "Shipping", "Refunds", "API", "Mobile app", "Reporting", "Integrations"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
SENTIMENTS = ["angry", "frustrated", "neutral", "positive"]
ADJ = ["Bold", "Quiet", "Rapid", "Golden", "Summer", "Winter", "Bright", "Prime", "Urban", "Fresh"]
NOUN = ["Falcon", "Harbor", "Cedar", "Comet", "River", "Wave", "Pulse", "Echo", "Summit", "Orbit"]
SUBJECTS = ["Refund not received for invoice", "Cannot log in after password reset", "Package delayed at depot",
            "API returns 500 on export", "App crashes on startup", "Report totals do not match",
            "Charged twice this month", "Integration token expired"]


# ----------------------------------------------------------------------------- number rendering
def _digits(s: str) -> str:
    return " ".join(ch for ch in s if ch != ",")  # "$1,234.56" -> "$ 1 2 3 4 . 5 6"


def money(x: float, style: str) -> str:
    s = "${:,.2f}".format(x)
    return _digits(s) if style == "digits" else s


def num(x: int, style: str) -> str:
    s = "{:,}".format(x)
    return _digits(s) if style == "digits" else s


def pct_(x: float, style: str) -> str:
    s = "{:.2f}%".format(x)
    return _digits(s) if style == "digits" else s


# ----------------------------------------------------------------------------- records
def service_record(rng: random.Random) -> dict:
    budget = rng.randrange(20, 1001, 10)
    daily = [round(rng.uniform(0.05, 2.0) * budget, 2) for _ in DAYS]
    requests = rng.randint(2_000, 500_000)
    err = round(rng.uniform(0.2, 5.0), 2)
    # last week's error rate differs by at least ~0.1 points, so "is it falling?" is never a tie.
    # (An earlier rejection loop, err * U(0.6, 1.4), never terminated for err < 0.25.)
    delta = rng.uniform(0.1, max(0.15, 0.4 * err))
    err_prev = round(err + delta if (rng.random() < 0.5 or err - delta < 0.05) else err - delta, 2)
    alerts = 0 if rng.random() < 0.5 else rng.randint(1, 200)
    return {"kind": "service", "service": "%s-%s-%d" % (rng.choice(ADJ), rng.choice(NOUN), rng.randint(1, 99)),
            "region": rng.choice(REGIONS), "country": rng.choice(COUNTRIES),
            "tier": rng.choice(TIERS), "status": rng.choice(STATUSES), "budget": budget,
            "daily": daily, "spend": round(sum(daily), 2), "requests": requests,
            "errors": int(round(requests * err / 100)), "alerts": alerts,
            "err": err, "err_prev": err_prev}


def ticket_record(rng: random.Random) -> dict:
    return {"kind": "ticket", "id": rng.randint(10_000, 99_999),
            "customer": "%s %s" % (rng.choice(ADJ), rng.choice(["GmbH", "Ltd", "Inc", "SA", "BV"])),
            "plan": rng.choice(PLANS), "channel": rng.choice(CHANNELS), "product": rng.choice(PRODUCTS),
            "priority": rng.choice(PRIORITIES), "assignee": rng.choice(AGENTS),
            "sentiment": rng.choice(SENTIMENTS), "days_open": rng.randint(0, 30),
            "sla_hours": rng.randint(1, 72), "replies": rng.randint(0, 25),
            "subject": "%s %s-%d" % (rng.choice(SUBJECTS), rng.choice(["INV", "ORD", "REQ"]), rng.randint(100, 9999))}


def render(r: dict, style: str = "raw") -> str:
    if r["kind"] == "service":
        lines = [
            "service: %s" % r["service"],
            "region: %s | country: %s | tier: %s | status: %s" % (r["region"], r["country"],
                                                                 r["tier"], r["status"]),
            "daily budget: %s" % money(r["budget"], style),
            "spend by day: " + ", ".join("%s %s" % (d, money(v, style)) for d, v in zip(DAYS, r["daily"])),
            "total spend: %s | requests: %s | errors: %s | alerts: %s" % (
                money(r["spend"], style), num(r["requests"], style), num(r["errors"], style),
                num(r["alerts"], style)),
            "error rate this week: %s | error rate last week: %s" % (pct_(r["err"], style),
                                                                    pct_(r["err_prev"], style)),
        ]
        if style == "derived":
            d = r["err"] - r["err_prev"]
            top = DAYS[max(range(7), key=lambda i: r["daily"][i])]
            lines.append("derived: error-rate change %+.2f points (%s) | spend is %.1fx the weekly budget | "
                         "highest-spend day %s | zero alerts: %s" % (
                             d, "falling" if d < 0 else "rising", r["spend"] / (7 * r["budget"]), top,
                             "yes" if r["alerts"] == 0 else "no"))
        return "\n".join(lines)
    lines = [
        "ticket #%d from %s (%s plan) via %s" % (r["id"], r["customer"], r["plan"], r["channel"]),
        "subject: %s" % r["subject"],
        "product: %s | priority: %s | assignee: %s | sentiment: %s" % (r["product"], r["priority"],
                                                                      r["assignee"], r["sentiment"]),
        "opened %s days ago | SLA hours left: %s | replies: %s" % (
            num(r["days_open"], style), num(r["sla_hours"], style), num(r["replies"], style)),
    ]
    if style == "derived":
        lines.append("derived: open more than a week: %s | SLA under 12 hours: %s" % (
            "yes" if r["days_open"] > 7 else "no", "yes" if r["sla_hours"] < 12 else "no"))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- questions
def _choice(rng, text, pool, answer, family, template, n_min=2, n_max=None):
    n_max = min(n_max or len(pool), len(pool))
    n = rng.randint(n_min, n_max)
    opts = [answer] + rng.sample([p for p in pool if p != answer], n - 1)
    rng.shuffle(opts)
    return {"text": text, "options": opts, "answer": opts.index(answer), "family": family, "template": template}


def _yesno(rng, text, truth, family, template):
    opts = ["yes", "no"] if rng.random() < 0.5 else ["no", "yes"]
    return {"text": text, "options": opts, "answer": opts.index("yes" if truth else "no"),
            "family": family, "template": template}


def _threshold(rng, value, lo=0.5, hi=1.5, gap=0.05):
    """A threshold that is clearly above or below `value` (50/50)."""
    if rng.random() < 0.5:
        return value * rng.uniform(lo, 1 - gap)
    return value * rng.uniform(1 + gap, hi)


def service_questions(r: dict, rng: random.Random, style: str = "raw") -> list[dict]:
    qs = [
        _choice(rng, "Which region is this service running in?", REGIONS, r["region"], "lookup", "region"),
        _choice(rng, "Which country hosts the service?", COUNTRIES, r["country"], "lookup", "country"),
        _choice(rng, "What is the service status?", STATUSES, r["status"], "lookup", "status"),
        _choice(rng, "What is the service tier?", TIERS, r["tier"], "lookup", "tier"),
    ]
    top = DAYS[max(range(7), key=lambda i: r["daily"][i])]
    qs.append({"text": "On which day was spend highest?", "options": rng.sample(DAYS, 7), "answer": None,
               "family": "numeric", "template": "peak_day"})
    qs[-1]["answer"] = qs[-1]["options"].index(top)
    # spend bucket: n contiguous buckets of equal width, one of which contains total spend
    n = rng.randint(3, 30)
    width = max(50, int(round(r["spend"] / rng.uniform(2, 8) / 50.0)) * 50)
    k = min(int(r["spend"] // width), 10_000)
    first = max(0, k - rng.randint(0, n - 1))
    edges = [(first + i) * width for i in range(n)]
    if not (edges[0] <= r["spend"] < edges[-1] + width):
        first = k
        edges = [(first + i) * width for i in range(n)]
    opts = ["%s to %s" % (money(e, style), money(e + width, style)) for e in edges]
    qs.append({"text": "Which range contains total spend?", "options": opts,
               "answer": [i for i, e in enumerate(edges) if e <= r["spend"] < e + width][0],
               "family": "numeric", "template": "spend_bucket"})
    qs.append(_yesno(rng, "Is the error rate falling compared with last week?", r["err"] < r["err_prev"],
                     "numeric", "err_falling"))
    x = round(_threshold(rng, r["spend"], 0.4, 1.6, 0.08), -1)
    qs.append(_yesno(rng, "Did total spend exceed %s with zero alerts?" % money(x, style),
                     r["spend"] > x and r["alerts"] == 0, "numeric", "spend_zero_alerts"))
    b = round(_threshold(rng, r["budget"], 0.4, 1.6, 0.1))
    qs.append(_yesno(rng, "Is the daily budget above %s?" % money(b, style), r["budget"] > b, "numeric",
                     "budget_above"))
    s = r["status"] if rng.random() < 0.5 else rng.choice(STATUSES)
    qs.append(_yesno(rng, "Is the service %s?" % s, r["status"] == s, "lookup", "status_is"))
    return qs


def ticket_questions(r: dict, rng: random.Random, style: str = "raw") -> list[dict]:
    qs = [
        _choice(rng, "Which product area is this ticket about?", PRODUCTS, r["product"], "lookup", "product"),
        _choice(rng, "Who is the ticket assigned to?", AGENTS, r["assignee"], "lookup", "assignee"),
        _choice(rng, "What is the ticket priority?", PRIORITIES, r["priority"], "lookup", "priority"),
        _choice(rng, "Which plan is the customer on?", PLANS, r["plan"], "lookup", "plan"),
        _choice(rng, "What is the customer's sentiment?", SENTIMENTS, r["sentiment"], "lookup", "sentiment"),
    ]
    p = r["plan"] if rng.random() < 0.5 else rng.choice(PLANS)
    qs.append(_yesno(rng, "Is the customer on the %s plan?" % p, r["plan"] == p, "lookup", "plan_is"))
    h = rng.randint(2, 72)
    while h == r["sla_hours"]:
        h = rng.randint(2, 72)
    qs.append(_yesno(rng, "Are fewer than %s hours left on the SLA?" % num(h, style), r["sla_hours"] < h,
                     "numeric", "sla_under"))
    d = rng.randint(1, 30)
    while d == r["days_open"]:
        d = rng.randint(1, 30)
    qs.append(_yesno(rng, "Has the ticket been open for more than %s days?" % num(d, style),
                     r["days_open"] > d, "numeric", "open_over"))
    return qs


def make_example(rng: random.Random, style: str = "raw") -> dict:
    if rng.random() < 0.5:
        r = service_record(rng)
        qs = service_questions(r, rng, style)
    else:
        r = ticket_record(rng)
        qs = ticket_questions(r, rng, style)
    return {"state": render(r, style), "questions": qs, "kind": r["kind"]}


def make_dataset(n: int, seed: int, style: str = "raw") -> list[dict]:
    rng = random.Random(seed)
    return [make_example(rng, style) for _ in range(n)]


# ----------------------------------------------------------------------------- long states (timing)
def long_state_ids(tok, n_tokens: int, seed: int = 0) -> list[int]:
    """Exactly n_tokens state tokens: concatenated service/ticket records, truncated."""
    rng = random.Random(seed)
    parts, ids = [], []
    while len(ids) < n_tokens:
        for _ in range(8):
            r = service_record(rng) if rng.random() < 0.6 else ticket_record(rng)
            parts.append(render(r))
        ids = tok("\n---\n".join(parts), add_special_tokens=False)["input_ids"]
    return ids[:n_tokens]


def question_pool(k: int, seed: int = 0) -> list[dict]:
    """k questions with a realistic mix of option counts (about half yes/no), for timing/exactness."""
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        ex = make_example(rng)
        out.extend(ex["questions"])
    rng.shuffle(out)
    return out[:k]
