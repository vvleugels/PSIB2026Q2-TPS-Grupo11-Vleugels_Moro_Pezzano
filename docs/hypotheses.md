# Hypotheses: resting-state EEG after sleep deprivation (SD) vs normal sleep (NS)

Pre-specified, testable hypotheses. The machine-readable version (used by the figures and the
summary table) is `analysis/hypotheses.py`; keep both in sync.

* **Contrast:** paired SD − NS within participant.
* **Measure:** mean log10 absolute band power per region (`results/features_region.csv`).
* **Analyses:** eyes open (EO) is primary and eyes closed (EC) secondary. EO and EC are never pooled.
* **Decision rule:** a hypothesis cell is *supported* when the effect has the expected sign and
  the BH-FDR-corrected Wilcoxon p < 0.05 (FDR across the 20 band × region tests of that eye
  condition). A significant effect in the other direction is reported as *opposite*.
* **Topography:** "largest over region X" is checked by whether the region with the largest
  |Cohen's dz| for that band is one of the hypothesised regions (`largest_in_expected_regions`).
* **Scope:** these are group-level spectral differences only; no clinical interpretation is
  intended.

| ID | Band | Regions (largest effect expected) | Expected direction | Eyes | Status |
|---|---|---|---|---|---|
| H1 | delta, theta | frontal, central (incl. fronto-central FC*) | increase (SD > NS) | EO and EC | confirmatory |
| H2 | alpha | parietal, occipital | decrease (SD < NS) | clearer in EC than EO | confirmatory, literature mixed |
| H3 | frontal theta/beta ratio | frontal | increase | EO and EC | confirmatory (single test per eye condition) |
| H4 | beta | all regions | no prediction | EO and EC | exploratory |

## H1: slow-wave activity

Delta (0.5–4 Hz) and theta (4–8 Hz) power increase after SD, with the largest increase over
frontal / fronto-central regions.

* References: TODO (authors, year, DOI), to be verified before filling in.

## H2: alpha

Alpha (8–13 Hz) power decreases after SD, with the largest decrease over posterior (parietal,
occipital) regions. The effect is expected to be clearer with eyes closed, where resting alpha
is dominant. **The literature is mixed:** both decreases and increases of resting alpha after
sleep loss have been reported, so a null or opposite result is a plausible outcome and is
reported as observed.

* References: TODO (authors, year, DOI), to be verified before filling in.

## H3: frontal theta/beta ratio

The ratio of theta to beta power over frontal channels increases after SD. It is computed as
the mean log10(theta/beta) over frontal channels and tested once per eye condition, without
FDR correction.

* References: TODO (authors, year, DOI), to be verified before filling in.

## H4: beta (exploratory)

Beta (13–30 Hz) power is examined in every region without a directional prediction. Results are
reported with FDR correction but interpreted as exploratory.

* References: TODO (authors, year, DOI), to be verified before filling in.
