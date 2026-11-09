"""Pre-specified hypotheses (band, region, expected direction) and their evaluation.

The text version, with rationale, is analysis/hypotheses.md. References are left as TODO on
purpose: they must be filled in only after the papers have been verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analysis.regions import REGIONS

ALPHA = 0.05
INCREASE, DECREASE, NONE = '+', '-', None


@dataclass(frozen=True)
class Reference:
    """A citation to be verified; every field stays 'TODO' until checked."""
    authors: str = 'TODO'
    year: str = 'TODO'
    doi: str = 'TODO'


@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    bands: tuple[str, ...]
    regions: tuple[str, ...]            # regions where the effect is expected to be largest
    direction: str | None               # '+' SD > NS, '-' SD < NS, None = no prediction
    measure: str = 'log10_power'        # or 'log10_frontal_theta_beta'
    eyes_focus: str = 'both'            # where the effect is expected to be clearest
    exploratory: bool = False
    note: str = ''
    references: tuple[Reference, ...] = field(default_factory=lambda: (Reference(),))


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        'H1', 'Delta and theta power increase after SD, largest over frontal / fronto-central regions.',
        bands=('delta', 'theta'), regions=('frontal', 'central'), direction=INCREASE,
        note='Fronto-central sites (FC*) belong to the "central" region in analysis/regions.py.'),
    Hypothesis(
        'H2', 'Alpha power decreases after SD, largest over posterior regions; expected clearer '
              'with eyes closed than eyes open. Prior findings are mixed (both decreases and '
              'increases of alpha after SD have been reported).',
        bands=('alpha',), regions=('parietal', 'occipital'), direction=DECREASE, eyes_focus='closed',
        note='Mixed literature: a null or opposite result is a plausible outcome, not an anomaly.'),
    Hypothesis(
        'H3', 'The frontal theta/beta ratio increases after SD.',
        bands=('theta/beta',), regions=('frontal',), direction=INCREASE,
        measure='log10_frontal_theta_beta'),
    Hypothesis(
        'H4', 'Beta power: exploratory, no directional prediction.',
        bands=('beta',), regions=tuple(REGIONS), direction=NONE, exploratory=True),
)


def direction_of(mean_diff: float) -> str:
    if pd.isna(mean_diff):
        return 'n/a'
    return 'increase' if mean_diff > 0 else 'decrease' if mean_diff < 0 else 'no change'


def agreement(expected: str | None, mean_diff: float, p_corrected: float) -> str:
    """Compare the observed paired effect with the expected direction.

    'supported' needs both the expected sign and p_corrected < 0.05; a significant effect in
    the other direction is reported as 'opposite', anything else as 'not significant'."""
    if pd.isna(p_corrected):
        return 'not tested'
    significant = p_corrected < ALPHA
    if expected is None:
        return 'exploratory: ' + ('significant' if significant else 'not significant')
    sign = '+' if mean_diff > 0 else '-'
    if not significant:
        return 'not significant'
    return 'supported' if sign == expected else 'opposite'


def evaluate(region_stats: pd.DataFrame, tbr_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per hypothesis x band x region x eyes with observed effects and agreement.

    ``largest_region`` tells whether the strongest effect (largest |dz|) of that band is in
    one of the hypothesised regions, which is the topographic part of H1/H2."""
    rows = []
    for h in HYPOTHESES:
        for eyes in ('open', 'closed'):
            if h.measure == 'log10_frontal_theta_beta':
                stats_rows = tbr_stats[tbr_stats['eyes'] == eyes]
                for _, s in stats_rows.iterrows():
                    # A single pre-specified test per eye condition: its p is not FDR-corrected.
                    rows.append(_row(h, eyes, 'theta/beta', 'frontal', s, s['p_wilcoxon'],
                                     'uncorrected (single test)', ''))
                continue
            for band in h.bands:
                band_rows = region_stats[(region_stats['eyes'] == eyes) & (region_stats['band'] == band)]
                if band_rows.empty:
                    continue
                strongest = band_rows.loc[band_rows['cohens_dz'].abs().idxmax(), 'region'] \
                    if band_rows['cohens_dz'].notna().any() else ''
                for region in h.regions:
                    s = band_rows[band_rows['region'] == region]
                    if s.empty:
                        continue
                    rows.append(_row(h, eyes, band, region, s.iloc[0], s.iloc[0]['p_fdr_wilcoxon'],
                                     'BH-FDR (20 band x region tests)', strongest))
    return pd.DataFrame(rows)


def _row(h: Hypothesis, eyes: str, band: str, region: str, s: pd.Series, p: float,
         correction: str, strongest: str) -> dict:
    expected = {'+': 'increase', '-': 'decrease', None: 'none (exploratory)'}[h.direction]
    return dict(hypothesis=h.id, eyes=eyes, analysis='primary' if eyes == 'open' else 'secondary',
                eyes_focus=h.eyes_focus, band=band, region=region, expected=expected,
                observed=direction_of(s['mean_diff']), n=int(s['n']), mean_diff=s['mean_diff'],
                ci95_low=s['ci95_low'], ci95_high=s['ci95_high'], p_corrected=p, correction=correction,
                rank_biserial=s['rank_biserial'], cohens_dz=s['cohens_dz'],
                agreement=agreement(h.direction, s['mean_diff'], p),
                largest_region=strongest,
                largest_in_expected_regions=(strongest in h.regions) if strongest else '')
