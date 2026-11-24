# All-station score

This note is separate from outputs/answers.md and the sprint deck.
The score uses hour, month, and station at all 18 stations. It does not use temperature or rainfall.

Reference case for the weights: hour 1, month 1, station Causeway Bay.
Largest upward weights: cat__hour_17 +2.48, cat__hour_18 +2.45, cat__hour_16 +2.37.
Largest downward weights: cat__hour_7 -2.53, cat__hour_6 -2.52, cat__hour_8 -2.50.
Largest upward station weights versus the reference station: Tap Mun +1.06, Tuen Mun +0.64, Yuen Long +0.44.
Largest downward station weights: Tseung Kwan O -0.73, Southern -0.49, Kwun Tong -0.47.
On 2024–2025 the same cutoff catches the most poor hours at Yuen Long TPR 88.2% FPR 34.3%, Tap Mun TPR 86.1% FPR 41.8%, Tuen Mun TPR 83.7% FPR 36.4%. Those stations also spend more safe hours indoors.

Stay in when the probability is at least 0.0241.
That cutoff was chosen on 2021–2023: the highest share of AQHI ≥ 7 hours caught with a false-positive rate at or under 20%.
Fit years: TPR 77.1%, FPR 19.8%, outdoors 79.2%.
2024–2025: TPR 63.3%, FPR 20.5%, outdoors 79.0%.
Old month-and-hour flags on the same hours: TPR 97.2%, FPR 61.0%, outdoors 38.6%.

Scores 1–5 are the hours below that cutoff, split into five equal training groups. 1 is the safest. 5 is the last go-outside score.
Scores 6–10 are the hours at or above the cutoff. 6 is the first stay-in score. 10 is the highest risk.
The bin edges were frozen on 2021–2023 and applied to 2024–2025.
1–5 go outside, 6–10 stay in.

Poor-hour share on 2024–2025 by score: 1 0.0% (n=46,144), 2 0.5% (n=46,708), 3 0.5% (n=46,606), 4 0.6% (n=46,569), 5 1.5% (n=46,802), 6 2.9% (n=12,306), 7 3.2% (n=12,194), 8 4.2% (n=12,394), 9 4.3% (n=12,418), 10 5.2% (n=12,710).
