> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

# Natural-rate prior conflict and the maternal-age gradient

**Original date:** 21 June 2026. Historical research note, revised for clarity in October 2026.

The early selection model produced large totals while moving the natural-rate parameters far from their Morris-based priors. The following tables record the prior conflict and the age pattern that prompted a revision.

| Variant | total_true | agg η | reduction (1−η) | agg s | max r̂ | min ESS |
|---|---|---|---|---|---|---|
| A (tight s) | 225,999 [208,870–242,706] | 0.855 | 0.145 | 0.067 | 1.01 | 417 |
| B (tight η_term) | 247,382 [225,508–269,935] | 0.936 | 0.064 | 0.062 | 1.02 | 82 |
| C (default) | 235,912 [219,586–254,246] | 0.892 | 0.108 | 0.064 | 1.01 | 215 |

| age | Morris prior | posterior | (post−prior)/σ |
|---|---|---|---|
| <20 | 0.00066 | 0.00197 | +10.9σ |
| 20–24 | 0.00070 | 0.00301 | +14.6σ |
| 25–29 | 0.00084 | 0.00350 | +14.3σ |
| 30–34 | 0.00148 | 0.00548 | +13.1σ |
| 35–39 | 0.00472 | 0.01749 | +13.2σ |
| 40–44 | 0.01522 | 0.05249 | +12.8σ |
| 45+ | 0.03071 | 0.05324 | +5.7σ |

| age | N frac | R | R/N | Morris | (R/N)/Morris = η·s |
|---|---|---|---|---|---|
| <20 | 0.045 | 433 | 2.89e-4 | 6.6e-4 | 0.438 |
| 20–24 | 0.184 | 1638 | 2.65e-4 | 7.0e-4 | 0.379 |
| 25–29 | 0.283 | 2613 | 2.76e-4 | 8.4e-4 | 0.328 |
| 30–34 | 0.295 | 3652 | 3.69e-4 | 1.48e-3 | 0.249 |
| 35–39 | 0.156 | 5326 | 1.02e-3 | 4.72e-3 | 0.216 |
| 40–44 | 0.034 | 3711 | 3.30e-3 | 1.52e-2 | 0.217 |
| 45+ | 0.003 | 403 | 4.54e-3 | 3.07e-2 | 0.148 |

| <20 | 20–24 | 25–29 | 30–34 | 35–39 | 40–44 | 45+ |
|---|---|---|---|---|---|---|
| ~0.99 | ~0.86 | ~0.75 | ~0.57 | ~0.49 | ~0.49 | ~0.34 |

| variant | total true DS 2016–24 | agg η | reduction | agg s |
|---|---|---|---|---|
| A | 40,202 [39,180–41,247] | 0.555 | 0.445 | 0.380 |
| B | 39,313 [36,810–41,914] | 0.543 | 0.457 | 0.386 |
| C | 39,110 [37,536–40,654] | 0.540 | 0.460 | 0.388 |

The observed rate relative to a natural-rate benchmark measures a combination of prenatal reduction and certificate recording, conditional on that benchmark and false-positive assumptions. A lower ratio at older ages does not by itself establish higher termination. Age-dependent recording could also contribute.

The revision tightened the natural-rate prior and added age terms to the reduction model. A tighter prior can keep a parameter close to its intended value. It can also transfer model conflict into other parameters. Check predictive fit and prior sensitivity rather than treating a stable natural-rate parameter as proof of a correct decomposition.

These estimates predate the surveillance anchor and current priors. The old claim that Boulet validated recording near 0.40 has been removed. See the [source review](20260804-salemi-boulet-study-area-transport.md) and [July diagnostic review](20260707-s-anchor-and-identifiability-diagnostic.md).
