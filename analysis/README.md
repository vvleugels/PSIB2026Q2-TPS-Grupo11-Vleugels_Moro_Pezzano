# Region features and SD vs NS statistics

Group-level comparison of resting-state band power after sleep deprivation (SD) vs normal
sleep (NS). Inputs are the batch outputs in `results/` (see [batch/README.md](../batch/README.md)).
These are **group-level spectral differences only**; nothing here supports clinical claims.

## Run

```bash
pip install -r requirements-analysis.txt      # pandas + Jupyter (the GUI does not need them)
python -m analysis.stats                      # all tables, ~20 s
python -m analysis.stats --no-clusters        # skip the per-electrode cluster test
python -m analysis.stats --permutations 10000
```

Or open `notebooks/03_stats.ipynb`, which calls the same functions and shows every table.

## Design

* **Primary:** eyes open (EO), every participant with both sessions.
  **Secondary:** eyes closed (EC), only participants who have both EC sessions.
  EO and EC are never pooled.
* **Paired, within subject:** difference = SD − NS for the same participant. Participants
  missing a session (e.g. a truncated recording) are left out of that analysis and listed
  in `stats_participants.csv`.
* **Epoch counts are not equalised** between sessions; they are reported in
  `stats_epochs.csv` (SD has fewer clean epochs than NS in EO).

## Region features (`analysis/regions.py`)

The 61 channel names come from the BIDS `channels.tsv` (identical in all 218 recordings) and
are grouped by their 10-10 prefix: frontal (Fp/AF/F, 16), central (FC/C/CP, 20),
temporal (FT/T/TP, 8), parietal (P, 9), occipital (PO/O, 8).

`features_region.csv` holds, per subject, session, eyes, band and region, the **mean log10
absolute power** (log of the geometric mean) and the **mean relative power** over the
region's channels, with the number of channels used. A region value is only computed when
at least **50 % of its channels** survived screening, because bad channels (mostly frontal,
from blinks) would otherwise leave regions represented by a few electrodes.

## Statistics (`analysis/stats.py`)

| Output | Content |
|---|---|
| `stats_region_log_power.csv` | per eyes × band × region: n, means, mean/median SD−NS, 95 % CI of the mean difference, Cohen's dz, rank-biserial r, Wilcoxon W and p, paired t and p, BH-FDR for both |
| `stats_region_rel_power.csv` | the same on relative power (%) |
| `stats_frontal_theta_beta.csv` | frontal theta/beta (mean log10 ratio over frontal channels), same reporting |
| `stats_session_order.csv` | SD−NS difference in NS→SD vs SD→NS participants (Mann-Whitney U, FDR) |
| `stats_epochs.csv` | clean/rejected epochs per session, paired test of clean epochs |
| `stats_participants.csv` | inclusion per analysis and the reason for each exclusion |
| `cluster_results.csv` | per eyes × band: each cluster's sign, channels, summed t, cluster p |
| `stats_cluster_channels.csv` | per electrode: t, mean SD−NS, cluster id/p, number of imputed values (for topoplots) |

* **Main test:** Wilcoxon signed-rank on SD−NS of log10 power. **Sensitivity:** paired t-test.
* **Effect sizes:** rank-biserial r = (W+ − W−)/(W+ + W−); Cohen's dz = mean(d)/sd(d).
  The 95 % CI is the t-based interval of the mean paired difference (log10 units:
  0.1 ≈ +26 % power).
* **Multiple comparisons:** BH-FDR across the 20 band × region tests, separately for EO and
  EC; uncorrected and corrected p-values are both reported. The frontal theta/beta test is a
  single pre-specified test per eye condition (not FDR-corrected).
* **Cluster test:** `mne.stats.permutation_cluster_1samp_test` on per-electrode SD−NS log10
  power, two-tailed, cluster-forming threshold t(n−1) at p < .05, 5000 sign-flip
  permutations (seed 0), adjacency from the dataset's own `electrodes.tsv` positions.

## Assumptions and limitations

* Screening is amplitude-only (not visual inspection); see the batch README.
* Bad channels differ between recordings. In the cluster test a missing electrode is set to a
  0 difference for that subject, which is conservative (pulls toward no effect); the count is
  in `n_imputed_zero`. Frontal and occipital regions have fewer participants than central.
* SD recordings have fewer clean epochs than NS. The averaged PSD is still unbiased, but noisier
  in SD; this was not equalised by design.
* Session order (counterbalanced) is checked with a between-group test, which has low power with
  ~30 participants per order group; a non-significant result does not prove there is no order effect.
* EC is a subgroup (~36 complete pairs) and is secondary; its results are not independent of EO
  (same participants).
