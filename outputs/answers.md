# Answers

Poor air means AQHI at or above 7 (EPD High, Very High, or Serious). Safer means AQHI 1–3 (Low). The bands are 1–3 Low, 4–6 Moderate, 7 High, 8–10 Very High, and 10+ Serious. Rates use 2021–2025 hours whose value is numeric and not marked with an asterisk.

## Question 1

- Claim: Poor hours follow the clock and the month. They rise through the afternoon and peak in September. They do not pile up at the roadside stations.
- Number: 1.6% of 742,343 station-hours have AQHI ≥ 7. The worst hour is 17 (5.4%). The worst month is Sep (4.7%); the calmest is Aug (0.6%). The highest year is 2022 (1.9%); the lowest is 2024 (1.1%).
- Chart: `outputs/figures/aqhi_by_hour.png`, `outputs/figures/aqhi_month_roadside.png`, `outputs/figures/aqhi_hour_month.png`.
- Roadside vs general: mean AQHI 3.49 at roadside versus 3.28 at general stations. Poor share 1.4% versus 1.6%. Low share 54.9% versus 61.8%. Highest region on the poor share: New Territories at 1.8%. Highest stations: Tap Mun 3.1%, Tuen Mun 3.0%, Yuen Long 2.3%.
- Limitation: 2026 is in the file and is not in these rates. Asterisk hours are excluded, so a station looks missing rather than copied from its neighbour. New Territories mixes Tap Mun, which has the most poor hours, with stations that are often Low.

## Question 2

- Claim: At Sham Shui Po, Kwun Tong, and Sha Tin, hotter days have a lower daily maximum AQHI. That link changes sign once the month is removed (Spearman 0.11). At the same stations, wetter hours have a lower AQHI. That hourly link disappears once the month is removed (Spearman -0.01). On days with at least 18 good gauge hours, a wetter day has a lower daily maximum AQHI. That daily link shrinks once the month is removed but does not disappear (Spearman -0.11).
- Number: Temperature Spearman -0.28 on 3,105 station-days (within month 0.11). Hourly rain Spearman -0.19 on 5,374 matched hours (within month -0.01). Daily rain, summed only on days with at least 18 QC = G hours, Spearman -0.45 on 216 station-days (within month -0.11).
- Chart: `outputs/figures/aqhi_vs_temperature.png`, `outputs/figures/aqhi_vs_rain.png`.
- Gauges: Sham Shui Po temperature SSP, rain gauge Tsung Tsin Christian Academy (6073, 0.1 km); Kwun Tong temperature KTG, rain gauge C.C.C. Mong Man Wai College (6139, 1.1 km); Sha Tin temperature SHA, rain gauge Sung Lan Middle School (6065, 0.9 km). Rain match covers 5,374 of 72,720 AQHI hours at these three stations, and 216 station-days clear the 18-hour gauge bar.
- Limitation: The project has no AQHI series. Rain is the nearest CoWIN gauge with QC = G, not the EPD inlet. A gauge day with fewer than 18 good hours is left out. Missing rain is not filled with zero.

## Question 3

- Claim: Go outside in the morning, and not in September or April. Poor means AQHI at or above 7. Hour 17 is the worst hour (5.4% poor) and September is the worst month (4.7%). August is the calmest month (0.6%). Roadside stations have fewer Low hours (54.9% versus 61.8% at general stations) and do not have more poor hours (1.4% versus 1.6%).
- Weather condition: do not wait for rain. Once the month is removed, the hourly rain link disappears. Hotter days in the year line up with a lower daily maximum AQHI because the hot months are cleaner. Within a month that temperature link is weak.
- Optional formula, three stations only: stay in at Sham Shui Po, Kwun Tong, or Sha Tin when the score is at least 0.009. That cutoff is the fit-year point with the highest share of poor hours caught whose false-positive rate is at or under 20%. It is not the citywide rule.
- Weather first, with no AQHI in the average: at those three stations the hottest month is Jul (32.7°C) and the coolest is Jan (20.6°C). Mean daily maximum is Sham Shui Po 27.9°C, Kwun Tong 27.0°C, Sha Tin 27.4°C. The wettest month is Sep (13.8 mm) and the driest is Dec (0.2 mm).
- Chart: `outputs/figures/hko_temp_by_month.png`, `outputs/figures/hko_rain_by_month.png`.
- Number: On the fit hours (2022: 3 hours, 2023: 23,593 hours) TPR is 94.2% and FPR is 19.2%. On 2024–2025 the false-positive rate stays at 19.5%, the share of poor hours caught falls to 43.9%, and he is sent out for 80.3% of hours.
- Caps, frozen on the fit years and read on 2024–2025: 10% (met): threshold 0.022, test TPR 28.7%, test FPR 9.8%, outdoors 90.0%; 20% (met): threshold 0.009, test TPR 43.9%, test FPR 19.5%, outdoors 80.3%; 30% (met): threshold 0.004, test TPR 54.8%, test FPR 29.9%, outdoors 69.9%.
- Same hours, old month-and-hour flags: test TPR 96.7%, test FPR 53.3%, outdoors 46.3%.
- 2024–2025 at this one threshold, not new cutoffs: DJF TPR 26.4% FPR 16.5%, MAM TPR 23.8% FPR 26.3%, JJA TPR 70.0% FPR 15.0%, SON TPR 85.3% FPR 19.7%. Sham Shui Po TPR 45.1% FPR 19.8%, Kwun Tong TPR 40.5% FPR 19.8%, Sha Tin TPR 46.0% FPR 18.8%.
- Formula: one logistic score on hour, month, station, same-hour temperature, and same-hour precipitation. After month and hour are in the score, a hotter hour raises the chance of AQHI ≥ 7 (log-odds +0.386) and a wetter hour lowers it (log-odds -0.333).
- Chart: `outputs/figures/aqhi_threshold_curve.png`.
- Limitation: Hourly temperature and rain start on 2023-01-01, so 2021 is absent and a 2022 date appears only when its hour 24 falls at that midnight. Open-Meteo is reanalysis at the HKO coordinates, not the EPD inlet and not an official rain gauge. The threshold was not chosen on 2024–2025. The fit-year catch rate does not carry over to those later years. 2026 is unused. The other 15 stations are not given this score.

## Graph gate

- Decision: no.
- Why: The file is one column per station. It has no coordinates and no edge list. Region and roadside are the scenario's labels.

## Project link

- The project did not score AQHI, so the rainstorm CSI numbers are not the comparison.
- What carries over: clocks in Hong Kong time, rainfall in millimetres, and a missing gauge hour is not filled with zero. The weather join uses the project window 2023-01-01 to 2025-12-31 and the station codes SSP, KTG, and SHA.

## Data handling

- Wide file: 49,225 rows, 18 stations, header after the EPD notes.
- Hourly cells: 850,608. Daily Max cells held out: 35,442.
- Blank or dash hourly cells: 9,918. Substituted hourly cells (asterisk): 22,873.
- Rates above use 742,343 numeric, non-substituted station-hours in 2021–2025. Low-band share overall is 60.6%.
