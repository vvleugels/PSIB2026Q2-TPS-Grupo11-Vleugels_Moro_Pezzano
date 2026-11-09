# Materials and Methods (draft)

> Draft generated from the code and `config.py`. Every number below is taken from the
> configuration or the dataset files; items marked **TODO** must be checked or completed by the
> authors before submission. No clinical claims are made: the analysis describes group-level
> spectral differences only.

## Dataset

We used the publicly available "A Resting-state EEG Dataset for Sleep Deprivation" (OpenNeuro
ds004902, version 1.0.8, doi:10.18112/openneuro.ds004902.v1.0.8; data descriptor: Xiang, Fan,
Bai, Lv & Lei, *Scientific Data* 11, 427, 2024, https://doi.org/10.1038/s41597-024-03268-2),
organised in BIDS 1.8.0 and released under CC0.

* **Participants:** 71 (sex and age in `participants.tsv`). TODO: report the sex distribution and
  mean ± SD age of the analysed sample, and the inclusion criteria from the data descriptor.
* **Design:** within-subject. Each participant was recorded after a night of normal sleep (NS,
  `ses-1`) and after sleep deprivation (SD, `ses-2`). Session numbers do not reflect time order:
  the order was counterbalanced and is given in the `SessionOrder` column of `participants.tsv`
  (NS→SD or SD→NS). TODO: describe the sleep-deprivation protocol (duration of wakefulness,
  supervision, time between sessions) from the data descriptor.
* **Recordings:** 5-min (300 s) resting-state EEG with eyes open for all participants; a subgroup
  (38 participants) also has eyes-closed recordings. TODO: confirm whether the eyes-closed condition
  is described as "partially" available in the descriptor, and the instructions given.
* **Acquisition (from the BIDS sidecars):** Brain Products GmbH system, 61 EEG electrodes in the
  10-10 layout, sampling rate 500 Hz, online reference FCz, power-line frequency 50 Hz,
  impedances below 5 kΩ (dataset README). TODO: amplifier model, ground electrode and online
  filters (the sidecar reports `SoftwareFilters: n/a`).

## Preprocessing

Processing was done in Python with MNE-Python, NumPy and SciPy (`batch/run_batch.py`; versions are
saved in `results/config_used.json`. TODO: copy the exact versions here). Original files were
never modified; the data were analysed in the original FCz reference, without re-referencing,
interpolation, ICA or resampling.

1. **File checks:** recordings whose binary `.fdt` file is shorter than its header requires were
   excluded. Five recordings were excluded for this reason (EO: sub-01 SD, sub-28 SD, sub-44 SD;
   EC: sub-03 SD, sub-28 NS).
2. **Filtering:** each continuous channel was band-pass filtered between 0.2 and 45 Hz before
   epoching, with a zero-phase (non-causal), one-pass FIR filter (windowed design, `firwin`,
   Hamming window, 0.0194 passband ripple, 53 dB stopband attenuation). Transition bandwidths
   were set automatically by MNE: 0.2 Hz at the lower edge (−6 dB at 0.1 Hz) and 11.25 Hz at the
   upper edge (−6 dB at 50.6 Hz), giving a filter length of 8251 samples (16.5 s). Signal edges
   were padded with `reflect_limited`. No notch filter was applied; because the upper −6 dB point
   lies above 50 Hz, line noise is only partly attenuated, which does not affect the analysed
   bands (≤ 30 Hz).
3. **Epoching:** the filtered signal was cut into consecutive, non-overlapping 5 s epochs starting
   at the first sample; the trailing incomplete epoch was discarded (300 s recordings → 60 epochs).
4. **Artifact screening:** for every epoch and each of the 61 channels, the peak-to-peak amplitude
   of the filtered signal was compared with fixed limits: > 150 µV (high amplitude) or < 1 µV
   (flat). These limits are starting values and were **not validated** against visual inspection
   for this dataset. A channel failing in more than 30 % of the complete epochs of a recording was
   treated as a bad channel for that recording (excluded from all features and not used to
   reject epochs). An epoch was rejected for all channels if any remaining channel failed it.
   Recordings with fewer than 12 clean epochs (60 s) were to be excluded; none reached this limit.
   Epoch counts were not equalised between sessions.

## Spectral features

* **Power spectral density:** Welch's method per clean epoch (2 s periodic Hann segments, 50 %
  overlap, mean of segments, density scaling, mean removed from each segment), giving a
  0.5 Hz resolution and four segments per epoch. Epoch spectra were averaged per channel and
  recording, and converted to µV²/Hz.
* **Band power:** absolute power was obtained by trapezoidal integration of the mean PSD over
  delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz) and beta (13–30 Hz), with linear
  interpolation at the exact band edges. Relative power was the band power divided by total
  0.5–30 Hz power (%). The theta/beta ratio was absolute theta divided by absolute beta power.
* **Regions:** channels were grouped by their 10-10 name into frontal (Fp/AF/F; 16 channels),
  central (FC/C/CP; 20), temporal (FT/T/TP; 8), parietal (P; 9) and occipital (PO/O; 8). Regional
  values are the mean of log10 absolute power (and of relative power) over the region's available
  channels, computed only when at least 50 % of the region's channels were valid. Frontal
  theta/beta was the mean log10 ratio over frontal channels.

## Statistical analysis

* **Analyses:** eyes open was the primary analysis (all participants with both sessions); eyes
  closed was secondary (only participants with both eyes-closed sessions). The two conditions were
  never pooled. Participants missing one session for a given condition were excluded from that
  analysis only.
* **Paired comparisons:** for each band × region, the within-subject difference SD − NS of log10
  power was tested with the Wilcoxon signed-rank test (two-sided, zero differences discarded).
  A paired t-test on the same log10 values served as a sensitivity analysis. Effect sizes:
  matched-pairs rank-biserial correlation and Cohen's d_z (mean difference / SD of differences),
  with the 95 % confidence interval of the mean difference (t distribution). The same tests were
  applied to relative power.
* **Multiple comparisons:** Benjamini–Hochberg false discovery rate across the 20 band × region
  tests, separately for each eye condition; both corrected and uncorrected p-values are reported.
  The frontal theta/beta ratio was a single pre-specified test per eye condition (uncorrected).
* **Session order:** the SD − NS differences of participants who completed NS first and those
  who completed SD first were compared with the Mann–Whitney U test (FDR across cells).
* **Electrode-level analysis:** per band and eye condition, SD − NS differences of log10 power at
  each electrode were tested with a cluster-based permutation test
  (`mne.stats.permutation_cluster_1samp_test`; two-tailed; cluster-forming threshold at the t
  value for p < .05 with n − 1 degrees of freedom; 5000 sign-flip permutations; cluster mass =
  sum of t; spatial adjacency by Delaunay triangulation of the dataset's electrode positions).
  A channel missing for a participant (bad channel in either session) entered as a zero
  difference, a conservative choice.
* **Hypotheses:** H1–H4 were specified before inspecting the results (`analysis/hypotheses.md`).
  TODO: confirm the date the hypotheses were fixed and add the verified references.
* **Significance level:** α = 0.05 (two-sided).

## TODO summary

* Sample description (sex, age, inclusion criteria) and the sleep-deprivation protocol.
* Amplifier model, ground electrode and online filters.
* Exact software versions (from `results/config_used.json` and `requirements*.txt`).
* Verified references for H1–H4 (`analysis/hypotheses.md`).
* Justification references for the 150 µV / 1 µV limits, the 30 % bad-channel rule and the 12-epoch
  minimum, or a statement that they are pragmatic choices.
