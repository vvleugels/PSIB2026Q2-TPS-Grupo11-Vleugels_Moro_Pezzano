# Batch band-power pipeline (no GUI)

Computes absolute and relative band power for **every subject, session (NS/SD), eye
condition and EEG channel** of ds004902, without opening the GUI. All settings come from
[`tps/config.py`](../tps/config.py), the single source of truth shared with the rest of the team.

Every computation step reuses the core modules in `tps/core/`, so a recording gives the
same numbers here and in the GUI with the same settings:

| Step | Function reused |
|---|---|
| BIDS inventory | `data.discover` |
| Loading + incomplete `.fdt` check | `data.open_recording` |
| FIR filter on the continuous channel | `processing.ContinuousFilter` (via `recording.filter.channel`) |
| 5 s epochs | `annotations.make_epochs` |
| Peak-to-peak screening | `screening.scan_recording` |
| Welch per epoch | `spectral.welch_epoch` |
| Band integration, relative power | `spectral.band_powers` |

## Run

From the project folder, with the project's virtual environment:

```bash
python -m tps.pipeline.batch                      # everything (both eye conditions)
python -m tps.pipeline.batch --eyes open          # only eyes open
python -m tps.pipeline.batch --subjects 01 02     # only some subjects (01 or sub-01)
python -m tps.pipeline.batch --jobs 6             # 6 recordings in parallel
python -m tps.pipeline.batch --force              # recompute finished recordings
python -m tps.pipeline.batch --subjects-file subsets/group20_seed2026.txt   # a fixed participant subset
python -m tps.pipeline.batch --rejection-mode per_channel   # each channel keeps its own clean epochs
```

On Windows: `.venv\Scripts\python.exe -m tps.pipeline.batch ...`. One recording takes about
10 s per core; the whole dataset (218 recordings) takes ~6 min with `--jobs 6`.

The run is **resumable**: each finished recording is cached in
`results/per_recording/<file>.json`, tagged with a hash of the configuration. Re-running
skips cached recordings unless the configuration changed (or `--force`). The final CSV
files are rebuilt from every cached recording, so `--eyes open` followed by
`--eyes closed` produces one combined table.

## Pipeline per recording

1. Skip and log recordings whose `.set`/`.fdt` is a git-annex link without content, or
   whose `.fdt` is shorter than its header requires.
2. Filter each whole continuous channel: 0.2–45 Hz zero-phase FIR (GUI default).
3. Cut 5 s non-overlapping epochs from sample 0; the trailing incomplete epoch is dropped.
4. Screen **all 61 EEG channels** on the filtered signal: a channel fails an epoch if its
   peak-to-peak amplitude is > 150 µV (high) or < 1 µV (flat).
5. **Bad channels:** a channel that fails in more than 30 % of the complete epochs is
   treated as bad for that recording: it gets no feature rows and does not reject epochs.
   Otherwise one noisy electrode (e.g. TP9 in sub-01 ses-1, failing 31/60 epochs) would
   discard half of the recording for all channels.
6. **Epoch rejection is global:** an epoch is rejected if any remaining channel fails it,
   so every channel of a recording uses the same epochs.
7. If fewer than 12 clean epochs (60 s) remain, the whole recording is excluded.
8. Welch per epoch (2 s Hann, 50 % overlap, density), averaged over clean epochs; band
   power is the integral of the mean PSD over each band. Relative power is relative to
   0.5–30 Hz.

## Outputs (`results/`)

| File | Content |
|---|---|
| `features_long.csv` | `sub, ses, condition (NS/SD), eyes (open/closed), channel, band, abs_power_uV2, rel_power_pct, n_epochs_used, n_epochs_rejected` |
| `theta_beta.csv` | `sub, ses, condition, eyes, channel, ratio` (absolute theta / absolute beta) |
| `exclusions.csv` | every excluded recording, bad channel, epoch and missing NS/SD counterpart, with the reason (`level` = recording / channel / epoch / pair) |
| `recordings_summary.csv` | one row per recording: excluded?, bad channels, epochs used/rejected |
| `session_order.csv` | `SessionOrder` (NS→SD or SD→NS) and EEG sampling times from `participants.tsv` |
| `config_used.json` | the exact configuration (and its hash) used for these outputs |
| `batch.log` | progress log of every run |

## Important: what "accepted" means

Automatic acceptance only means the epoch was **within the amplitude limits** on every
non-bad channel. It is **not** visual inspection: ocular, muscular or movement artifacts
below 150 µV p-p pass, and the 150 µV / 1 µV thresholds are starting values taken from the
GUI, **not validated** for this dataset. Blinks in eyes-open recordings often exceed
150 µV on frontal channels (Fp1, Fpz, AF3/AF4…), which then end up as "bad channels".
Check `exclusions.csv` before interpreting frontal results.

## Tests

```bash
python -m unittest discover -s tests -p "test_pipeline_batch.py" -v
```

The tests use synthetic signals only (a 10 Hz sine must put > 90 % of its power in alpha,
a 6 Hz + 20 Hz mix gives the expected theta/beta ratio, spikes reject exactly their epoch,
a persistently noisy channel becomes a bad channel, too few clean epochs exclude the
recording, git-annex placeholders are logged, and re-running skips finished recordings).
