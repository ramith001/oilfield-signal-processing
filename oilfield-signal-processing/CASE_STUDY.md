# Case study: looking for stick-slip in public drilling telemetry

Ramith Vaid, NTU Singapore. Numbers below come from the README and the tables in `data/`.

## The question

Stick-slip is a torsional oscillation of a drillstring in which the bit alternately sticks and spins free. It wears bits, damages tools and slows drilling. The Volve field data, released by Equinor, include real-time surface logs (weight on bit, torque, rotary speed, bit depth, hookload) sampled every 2 to 3 seconds. The question was whether stick-slip can be detected and graded from that surface data alone, and whether stronger stick-slip costs drilling speed.

## What was built

- A parser for the raw WITSML XML, including the column-order convention of the index curve, and cleaning for two sensor artifacts: torque held at a frozen value for long stretches (88.6% of active-drilling rows in one well sit in a 60 s window where torque never updates) and a 16-bit saturation ramp near 32,767 N.m.
- A spectral peak test that compares each window's strongest peak with a fitted red-noise background, with p-values from a parametric bootstrap that was itself calibrated on simulated noise.
- A rolling severity index, a spectrogram of the dominant period, and an automatic event detector with four severity labels, validated against the index, against phase-randomised copies of the data with the same spectrum, and against synthetic oscillations of known size.

## What the checks found

The first version reported a dominant torque period inside the expected 5 to 120 s band for two of three wells, and a positive correlation of severity with weight on bit in all three (Spearman 0.27, 0.37, 0.26). Each later check took something away.

1. **The odd period was an artifact.** The 2,911 s period reported for well 9-F-4 equals the segment length divided by three, the lowest frequency the search allowed. Cutting the segment shorter moved the period with the window (median 200 s in 10-minute windows, 400 s in 20-minute windows). No in-band period for that well was supported.
2. **The severity windows crossed gaps.** The index used 30 rows of a filtered subset, not 60 seconds, so 17% to 52% of windows reached across a gap, sometimes by hours. On time-contiguous windows the WOB correlations fell to 0.18, 0.29 and 0.08.
3. **The wells were not rotating.** Adding wells showed that 9-F-9, 9-F-7 and 9-F-4 have surface rotary speed above 30 rpm for 0.2, 0.1 and 0.1 hours of 33.7, 41.3 and 166.1 hours of record. The earlier results describe torque variation on a stationary string, and the rig activity code used to select "drilling" rows does not mean rotation in these logs. Six more wells that did drill (3 to 216 hours of rotating drilling each) came from a converted public copy of the same release.
4. **On rotating wells the old findings do not hold.** Relative to mean torque, severity is unrelated or negatively related to weight on bit (-0.41 to 0.00 across the six wells), and it falls with depth only because mean torque rises with depth. The strong depth result from 9-F-4 does not replicate. Rate of penetration shows no detectable link to severity once weight on bit and rotary speed are removed (-0.03 and -0.11 in the two wells with ROP data, intervals that include zero).
5. **Spectral peaks and detector events are not distinguishable from noise.** A steady period near 25 s appears in torque, weight on bit and hookload alike and does not rise with depth from 100 to 4,700 m. The detector's amplitude tracks the severity index (Spearman 0.37 to 0.84), but it finds regular events as often in phase-randomised noise as in the real torque (6.9 against 6.9 per hour in 9-F-9, 1.6 against 1.7 in 9-F-15b), and a noise-only synthetic series triggers one in 37% of cases at the noise level of the rotating wells.

## Why the data cannot answer the question

The sampling interval limits it. Rows arrive on a 3, 3, 3, 3, 2 s cycle (mean 2.8 s), so the shortest resolvable period is 5.6 s. A torsional fundamental of roughly 4 L / c, with a wave speed near 3,000 m/s, is about 1.3 to 6 s for strings of 1,000 to 4,700 m. That estimate puts most classic stick-slip below what this data can see, and the 25 s feature is something slower.

## What I would do next

Find rotating wells with surface data at 1 s or faster, or a downhole vibration measurement, and repeat the severity, spectral and detector steps there; compare ROP at matched depth intervals instead of pooled rank correlations; and use the synthetic test to set the lowest oscillation size a detector can grade before any label is used operationally.

## What I take from it

A first result that looks plausible (a period in the right band, a correlation with the right sign) is not a finding until it survives a noise null, a time-contiguous re-analysis and a check that the data contain the physics being asked about. Here the last check, whether the string was rotating, came last and changed the most. The repository keeps each correction next to the result it qualifies, so the path from the first version to the current reading can be audited.
