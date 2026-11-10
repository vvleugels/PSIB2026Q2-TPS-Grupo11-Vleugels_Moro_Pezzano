"""Pre-specified hypotheses (band, region, expected direction) and their evaluation.

The text version, with rationale, is docs/hypotheses.md. References are left as TODO on
purpose: they must be filled in only after the papers have been verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tps.analysis.regions import REGIONS

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
    exploratory: bool = False
    references: tuple[Reference, ...] = field(default_factory=lambda: (Reference(),))


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis('H1', 'La potencia delta y theta aumenta tras SD, con el mayor aumento en las regiones '
                     'frontal y central (incluye las fronto-centrales FC*).',
               bands=('delta', 'theta'), regions=('frontal', 'central'), direction=INCREASE),
    Hypothesis('H2', 'La potencia alfa disminuye tras SD, con la mayor disminución en las regiones '
                     'posteriores; se espera más clara con ojos cerrados. La literatura está dividida.',
               bands=('alpha',), regions=('parietal', 'occipital'), direction=DECREASE),
    Hypothesis('H3', 'El cociente theta/beta frontal aumenta tras SD.',
               bands=('theta/beta',), regions=('frontal',), direction=INCREASE),
    Hypothesis('H4', 'Potencia beta: exploratoria, sin predicción de dirección.',
               bands=('beta',), regions=tuple(REGIONS), direction=NONE, exploratory=True),
)


def direction_of(change: float) -> str:
    if pd.isna(change):
        return 'n/a'
    return 'increase' if change > 0 else 'decrease' if change < 0 else 'no change'


def agreement(expected: str | None, change: float, p: float) -> str:
    """'supported' needs the expected sign and p < 0.05; a significant effect in the other
    direction is 'opposite'; anything else 'not significant'."""
    if pd.isna(p):
        return 'not tested'
    significant = p < ALPHA
    if expected is None:
        return 'exploratory: ' + ('significant' if significant else 'not significant')
    if not significant:
        return 'not significant'
    return 'supported' if ('+' if change > 0 else '-') == expected else 'opposite'


def evaluate(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per hypothesis x eyes x band x region from stats_summary.csv.

    ``largest_region`` is the region with the largest |% change| for that band, to check the
    topographic part of H1/H2."""
    rows = []
    for h in HYPOTHESES:
        for eyes in ('open', 'closed'):
            for band in h.bands:
                band_rows = summary[(summary['eyes'] == eyes) & (summary['band'] == band)]
                if band_rows.empty:
                    continue
                changes = band_rows['change_pct'].abs()
                strongest = band_rows.loc[changes.idxmax(), 'region'] if changes.notna().any() else ''
                for region in h.regions:
                    s = band_rows[band_rows['region'] == region]
                    if s.empty:
                        continue
                    s = s.iloc[0]
                    rows.append(dict(
                        hypothesis=h.id, eyes=eyes, band=band, region=region,
                        expected={'+': 'increase', '-': 'decrease', None: 'none (exploratory)'}[h.direction],
                        observed=direction_of(s['change_pct']), n=int(s['n']),
                        change_pct=s['change_pct'], change_db=s['change_db'], p=s['p'],
                        agreement=agreement(h.direction, s['change_pct'], s['p']),
                        largest_region=strongest))
    return pd.DataFrame(rows)
