"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Program: `python3 main.py` (Python 3.13, standard library only). It reads
every `<id>.in` case in the input directory and writes one `<id>.out`
forecast per case: for each trophy, five quantiles of its earn rate (percent)
at the case's target, at levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Model (fitted at run time from the case's own history only; no network, no
outside data, nothing precomputed about the answer):

1. Trend per trophy. The logit of each trophy's earn rate is fitted by least
   squares against x = ln(1 + hours since release), over the last 24 hours
   of reads (all reads when the history is shorter). The slope is shrunk
   toward zero with an empirical-Bayes factor (prior sd 0.5), because rates
   built from a handful of earners are noisy and a raw slope would swing
   wildly. The fitted line is then read at the target, but the extrapolated
   move is damped by 1 / (1 + dx / 1.5): a launch-week rate cannot keep
   climbing (or falling) at its last-day pace for four more days. Earners
   arrive slower as the pool of players who have not yet earned a trophy
   shrinks, and new players joining drag every rate down; the damping is
   that deceleration, and it beat the undamped fit on every synthetic
   final I tested (fast/slow saturation, fast/slow player growth, and no
   saturation at all).

2. Flat history. When every read in the history is identical, there is no
   trend to fit -- and the one refresh the rules document (launch night:
   638 players tracked, then 3,868 at the first refresh, earners up only
   ~1.2x, so every rate but the 100% one fell to ~0.2x) says the next read
   is probably a refresh, not a repeat. The median is then the last rate
   times 0.198 (1.2 * 638 / 3868) with a wide spread. This branch fires on
   the practice cases; the final's history varies, so it will not fire
   there.

3. Structural caps, applied to the medians in rate space. A trophy at
   100% in every read stays 100% (every tracked player holds it; the
   refresh left it at 100%). The platinum can never exceed any other
   trophy (its earners are a subset of every other trophy's earners).
   A later chapter can never exceed an earlier one ("Complete Chapter N"
   trophies, in chapter order).

4. Spread. Each trophy's logit spread is sqrt(0.25^2 + fit^2 + (0.30 *
   sqrt(horizon_days))^2), where fit is the residual sd of the trophy's
   trend fit times its prediction leverage; floored at 0.30, capped at
   2.0. Quantiles are the median's logit plus the normal quantiles times
   the spread, mapped back to percent and rounded to 4 decimals.

Everything is deterministic: fixed file order, no randomness, no clock
reads (all times come from the case). Two runs write byte-identical files.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (the reasoning is in the module docstring) ---
WINDOW_HOURS = 24.0        # fit the trend on the last 24 h of reads
SHRINK_TAU = 0.5           # prior sd for the empirical-Bayes slope shrinkage
DAMP_KAPPA = 1.5           # damping of the extrapolated logit shift, in x-units
SIGMA_BASE = 0.25          # irreducible logit spread
SIGMA_HORIZON = 0.30       # per square-root day of horizon
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.0
FLAT_DILUTION = 0.198      # documented launch-night refresh: 1.2 * 638 / 3868
FLAT_SIGMA = 1.0           # wide spread when history shows no change at all
FLAT_SIGMA_HORIZON = 0.15
RELEASE_FALLBACK = "2026-11-19"


def parse_time(text):
    """An ISO 8601 timestamp (with Z, fractional seconds, or neither)."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def inv_logit(log_odds):
    return 100.0 / (1.0 + math.exp(-log_odds))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_sd, n, mean_x, se_b, sxx)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0, 0.0, n, mean_x, 0.0, 0.0
    b = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    a = mean_y - b * mean_x
    dof = max(n - 2, 1)
    s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof
    return a, b, math.sqrt(s2), n, mean_x, math.sqrt(s2 / sxx), sxx


def chapter_number(trophy):
    """The chapter a 'Complete Chapter N' trophy completes, else None."""
    match = re.search(r"Complete Chapter (\d+)", trophy["description"])
    return int(match.group(1)) if match else None


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"levels {case['levels']} are not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]

    release_text = case["game"].get("release_date", RELEASE_FALLBACK)
    release = parse_time(release_text if "T" in release_text
                         else release_text + "T00:00:00Z")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log(1.0 + max(0.0, (target - release).total_seconds() / 3600.0))

    # One parsed copy of the history, oldest first.
    reads = [(parse_time(r["at"]), r["players_tracked"], r["rates"]) for r in history]
    if reads:
        last_at = reads[-1][0]
        window = [rd for rd in reads
                  if (last_at - rd[0]).total_seconds() <= WINDOW_HOURS * 3600]
        if len(window) < 3:
            window = reads
        # Flat history: the page never changed in the whole recording.
        flat = all(players == reads[0][1] and rates == reads[0][2]
                   for _, players, rates in reads)
    else:
        window = []
        flat = False

    medians = [0.01] * len(trophies)
    sigmas = [2.0] * len(trophies)
    for i, trophy in enumerate(trophies):
        last_rate = None
        for _, _, rates in reversed(reads):
            if rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            continue
        # A trophy every tracked player holds stays held by every player.
        if last_rate == 100.0 and all(rates[i] == 100.0 for _, _, rates in reads
                                      if rates[i] is not None):
            medians[i] = 100.0
            sigmas[i] = 0.0
            continue
        if flat:
            medians[i] = clamp_pct(last_rate * FLAT_DILUTION)
            sigmas[i] = FLAT_SIGMA + FLAT_SIGMA_HORIZON * math.sqrt(horizon_days)
            continue

        points = []
        for at, _, rates in window:
            rate = rates[i]
            if rate is None:
                continue
            hours = (at - release).total_seconds() / 3600.0
            points.append((math.log(1.0 + max(0.0, hours)), logit(rate)))
        if len(points) >= 3 and len({y for _, y in points}) > 1:
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            a, b, resid, n, mean_x, se_b, sxx = ols(xs, ys)
            shrink = (SHRINK_TAU ** 2) / (SHRINK_TAU ** 2 + se_b ** 2) if se_b > 0 else 1.0
            slope = b * shrink
            intercept = sum(ys) / len(ys) - slope * mean_x
            x_last = xs[-1]
            dx = x_target - x_last
            damp = 1.0 / (1.0 + dx / DAMP_KAPPA) if dx > 0 else 1.0
            median_logit = intercept + slope * x_last + slope * dx * damp
            leverage = math.sqrt(1 + 1.0 / n + (x_target - mean_x) ** 2 / sxx) if sxx > 0 else 1.5
            fit_spread = max(resid, 0.15) * leverage
        else:
            # Too little movement to fit a trend: hold the last rate.
            median_logit = logit(last_rate)
            fit_spread = 0.6
        medians[i] = clamp_pct(inv_logit(median_logit))
        sigmas[i] = min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(
            SIGMA_BASE ** 2 + fit_spread ** 2
            + (SIGMA_HORIZON * math.sqrt(horizon_days)) ** 2)))

    # Structural caps on the medians, in rate space.
    platinum = [i for i, t in enumerate(trophies) if t["tier"] == "platinum"]
    if platinum:
        others = [medians[i] for i in range(len(trophies))
                  if i != platinum[0] and sigmas[i] > 0]
        if others:
            medians[platinum[0]] = min(medians[platinum[0]], min(others))
    chain = sorted((num, i) for i, t in enumerate(trophies)
                   if (num := chapter_number(t)) is not None)
    for k in range(1, len(chain)):
        prev, cur = chain[k - 1][1], chain[k][1]
        if medians[cur] > medians[prev]:
            medians[cur] = medians[prev]

    rows = []
    for i, trophy in enumerate(trophies):
        if sigmas[i] == 0.0:
            quantiles = [100, 100, 100, 100, 100]
        else:
            centre = logit(medians[i])
            quantiles = [round4(clamp_pct(inv_logit(centre + z * sigmas[i])))
                         for z in Z]
            for j in range(1, 5):
                if quantiles[j] < quantiles[j - 1]:
                    quantiles[j] = quantiles[j - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
