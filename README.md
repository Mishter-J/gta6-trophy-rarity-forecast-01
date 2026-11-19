# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mishter-J/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mishter-J/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**GTA VI Trophy Rarity Forecaster** — forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, its earn rate on Exophase one week after launch (2026-11-26T00:00:00Z), as five quantiles per trophy. Built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

## How it works

`main.py` (Python 3.13, standard library only) reads each case file in the input directory and writes one forecast file per case. The model is fitted at run time from the case's own history — no network, no outside data:

- **Per-trophy trend.** The logit of each trophy's earn rate is fitted by least squares against ln(1 + hours since release) over the last 24 hours of reads. Slopes are shrunk toward zero (empirical-Bayes, prior sd 0.5) because launch-night rates rest on a handful of earners. The extrapolated move is damped by 1 / (1 + dx / 1.5): a rate cannot keep its last-day pace for four more days — earners arrive slower as the un-earned pool shrinks, and new players joining drag every rate down.
- **Flat history.** When every read is identical, there is no trend to fit — and the one refresh the rules document (launch night: 638 players tracked, then 3,868 at the first refresh, earners up only ~1.2x, so every rate but the 100% one fell to ~0.2x) says the next read is probably a refresh. The median is then the last rate × 0.198 with a wide spread. This branch fires on the practice cases; the final's history varies, so it will not fire there.
- **Structural caps.** A trophy at 100% in every read stays 100%. The platinum never exceeds any other trophy. A later chapter never exceeds an earlier one.
- **Spread.** Each trophy's logit spread combines a base term, the trend fit's residual × prediction leverage, and a term growing with the square root of the horizon — so the five quantiles are honest about how far ahead the forecast reaches.

On the practice cases, scored against the refresh reconstructed from the rules (players 638 → 3,868, earners ×1.2), this program scores a mean pinball loss of about 0.20, against the house baseline's 1.51. The practice pays nothing and one 90-minute refresh is little guide to a week, so the number is a check that the program reads a case and copes with a refresh — not a promise about the final.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

That reads every `*.in` case in `input/` and writes `<id>.out` forecasts to `out/`. The grader runs `python3 /bot/main.py` with `/input` and `/output` instead. Output is one JSON object per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing, each 0–100.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
