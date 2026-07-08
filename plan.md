# Plan de travail — ML-Driven Dispersion Trading Strategy

> **Cadrage calendaire :** démarrage 25 juin 2026, objectif de fin 31 juillet 2026 → ~5 semaines. Rythme intensif. Le ML est une couche bonus, pas un prérequis.

---

## SEMAINE 1 (29 juin → 5 juil) — Setup, Données & Univers

**Objectif :** infrastructure prête + dataset propre point-in-time. C'est la semaine la plus critique.

### Setup
- [✅] Connexion WRDS opérationnelle (`wrds.Connection()`) — compte validé, `.pgpass` créé, accès `crsp`/`optionm`/`comp` confirmé
- [✅] `optionm.vsurfd` exploré (notebook 01) : tables annuelles 1996→2025, colonnes connues, 91j/delta±50 natifs, SPX = secid 108105
- [✅] CRSP exploré (notebook 02) : membership `crsp.dsp500list` (1925→2024), réalisé `crsp.dsf` (prc/ret/shrout), jointure `secid↔permno` = `wrdsapps_link_crsp_optionm.opcrsphist`. Produit moderne `idx_const_*_v2` = accès refusé → approche legacy retenue.
- [✅] Repo GitHub + environnement (uv + pyproject.toml : wrds, pandas, numpy, scipy, scikit-learn, statsmodels)
- [✅] Plomberie de connexion WRDS écrite (`src/dispersion/data/wrds_client.py` + `.env`)
- [✅] **Paramètres de design figés** : N=100 (top capi, rotation ~4/trimestre, couverture 63–70%), maturité 91j, delta ±50, rebalancement trimestriel. SPX = secid 108105.
- [✅] **Période de backtest figée : 1996 → 2024** (~29 ans, ~116 rebal.). Diagnostic couverture : 97–100/100 composantes avec IV + SPX OK chaque année dès 1996 ; borne haute = fin `dsf` (31/12/2024).

### Données
- [✅] CRSP : `get_universe(db, date, n=100)` → `src/dispersion/data/universe.py` — testé sur 2010/2015/2020, 0 secid manquant
- [✅] OptionMetrics `vsurfd` : IV pour SPX + composants, 91j, delta ±50 → `get_iv()` (`src/dispersion/data/iv.py`), testé mars 2020 : 0 trou, sanity diversification OK
- [✅] Jointures d'identifiants : `secid` ↔ `permno` via `opcrsphist` score=1, intégré dans `get_universe`
- [✅] CRSP `dsf` → rendements + vol réalisée + corrélation réalisée : `get_returns`, `realized_vol`, `realized_corr_matrix` (listwise), `average_correlation` (ρ̄ pondérée, option B) — `src/dispersion/data/returns.py`, testés COVID vs calme OK. Matrice singulière (T=63<N=100) → confirme l'intérêt RMT.
- [✅] Nettoyage : calendrier maître = intersection vsurfd(SPX)∩dsf ; IV ffill borné 3j ; outliers composantes hors percentiles 0.1/99.9 → NaN ; SPX non clippé. (`src/dispersion/data/assemble.py`)
- [✅] Assemblage `build_dataset(db, 1996, 2024)` → 6 `.parquet` dans `data/processed/` (build complet en 7 min).
- [✅] QA adversariale multi-agents (6 dimensions) : 4 PASS / 2 WARN. 1 seul vrai souci = trou OptionMetrics 27/07→18/08/2020 (panne fournisseur, IV nulle sur toute la surface) → accepté + documenté (README §9bis). **Semaine 1 TERMINÉE.**

**Livrable :** ✅ fichiers `.parquet` produits (`iv_index`, `iv_components`, `weights`, `realized_vol`, `realized_corr`, `corr_matrices`) alignés sur le calendrier maître. Reproductible via `build_dataset`.

**Pièges :** survivorship/look-ahead bias → point-in-time obligatoire · la jointure d'identifiants est le vrai goulot, attaque-la dès le jour 1.

---

## SEMAINE 2 (6 juil → 12 juil) — Théorie & Signal

**Objectif :** la mécanique du signal + sa justification théorique.

### Dérivation mathématique — ✅ acquise via lecture complète DMV (2009), 8 juil.
- [✅] P&L du portefeuille de dispersion via Lemme d'Itô — éq. (8)–(11) redérivées : hedge triangulaire (vega d'abord, la vol-de-vol se simplifie dans le ratio de vegas relatifs ; delta résiduel avec l'indice seul). Rédaction pour le rapport en S5.
- [✅] Décomposition en pari corrélation réalisée vs implicite — éq. (2)–(3), identification par VRP_indiv ≈ 0 (98/127 titres)
- [✅] Formalisation VRP + CRP — éq. (3) via Itô/Girsanov (convexités s'annulent en ℚ−ℙ) ; MFIV éq. (4)–(5) Carr–Madan comprise

### Forks méthodologiques tranchés (8 juil., README §5 + §8)
- [✅] **Variance implicite = ATM 91j** (instrument-consistency : on trade des straddles ATM ; biais conservateur : le skew indice fait sous-estimer ρ_implied) ; MFIV = robustesse S5 si temps
- [✅] **Window-matching = deux séries** : trailing (signal ex-ante, S3) + forward = trailing décalé de +63 j ouvrés (validation de la prime à la DMV) ; caveat poids inter-trimestre documenté
- [✅] **Micro-choix d'implémentation (8 juil., vérifiés sur données)** : IV manquante → renormalisation quotidienne des poids sur titres disponibles + plancher 90/100 (« supprimer les jours » aurait détruit 54,9 % de l'échantillon ; poids manquant médian 1,3 %) ; ρ_implied stocké **brut** + compteurs de violations ; livrable `signal.parquet` ; `secid` → Int64 (README §5.1)

### Corrélation implicite (implémentation)
- [✅] Formule d'inversion implémentée + validée (notebook 03, 8 juil.) → promue en `src/dispersion/signal/implied_corr.py`
- [✅] ρ_realized depuis la covariance réalisée (fait en S1 : ρ̄ pondérée quotidienne, option B — même forme fonctionnelle que l'inversion ⟹ spread pomme-à-pomme)
- [✅] Séries **premium** et **signal** calculées : Π̄ = **+0.079 (t-NW63 = 7.4, 75 % j > 0)**, S̄ = +0.078 (t = 10.6, 80 % j > 0) → **thèse CRP validée sur 1996–2024**
- [✅] Vérification bornes : ρ_implied ∈ [0.12, 0.91] — **0 violation, 0 NaN** sur 7221 j ; couverture min 91/100 (plancher jamais atteint)
- [✅] Aperçu sous-périodes : 2008–12 = 0.135 (t=4.3) ; **2020–24 = 0.028 (t=1.19, non significatif)** → compression récente de la prime, à creuser en S5 (limits-to-arbitrage)
- [✅] Promotion : `src/dispersion/signal/implied_corr.py` (`implied_correlation` + `build_signal`) + `signal.parquet` écrit (7221 j × 7 col.) — reproduit le notebook à l'identique.
- [✅] **Audit complet fin S2** (4 axes indépendants : données, signal, maths, cohérence) : 0 finding critique ; ρ_implied recalculé indépendamment (écart ~1e-16) ; corrections documentaires appliquées ; fragilités latentes → README §9bis. **SEMAINE 2 TERMINÉE** (rédaction des dérivations → S5).

**Livrable :** ✅ `signal.parquet` + graphe de validation `results/figures/fig_crp_validation.png` (implicite au-dessus du réalisé forward, premium window-matché).

**Piège :** approximation one-factor (corrélation moyenne unique) — à assumer et documenter (DMV éq. 6 fait la même hypothèse).

---

## SEMAINE 3 (13 juil → 19 juil) — Stratégie Baseline

**Objectif :** un backtest qui tourne, sans ML = benchmark de référence.

### Moteur
- [✅] `backtest/marking.py` (8 juil.) : interpolation en variance totale 30/60/91 + σ(30j) gelée, `RateCurve` (ffill borné, échec bruyant), `adjust_strike` (splits cfadj) — 7 tests pytest (17/17 au total)
- [✅] Classe `DispersionEngine` (8 juil., `backtest/engine.py`) : book short straddle indice / long straddles composants, entrée aux primes réelles + q̂ par parité C−P, sizing richesse λ (neutralité aux chocs PROPORTIONNELS de vol — seule convention qui reproduit le +101,12 % de DMV, car ν≈1/σ pour un straddle ATM), marquage quotidien, hedge delta indice quotidien (éq. 10–11), règlement intrinsèque, splits cfadj, v1 câblée (`threshold`) — **5 tests en forme fermée** (22/22 au total)
- [✅] **v0 inconditionnel EXÉCUTÉ (8 juil., 222 s, 115 trimestres) — benchmarks DMV AU RENDEZ-VOUS** : **Σyᵢ = 99,5 %** (eux +101,12 %), **Sharpe brut 0,77** (eux 0,73), 70 % trimestres > 0, skew −1,29 ; ret trim. moyen +7,3 % brut ; pires trimestres = **Q1-2018 Volmageddon (−91 %), Q3-2024 yen carry (−41 %), été 2002 (−30 %)** ; maxDD −95,8 % brut → l'argument central pour v1/ML (couper les queues). Compteurs : marks gelés 0,21 j/pos/trim ; 621 règlements au spot J-1 = rotation d'univers (~5/trim, fix au prochain rebuild). Sorties : `backtest_v0_{daily,quarterly}.parquet`
- [✅] **Mécanisme Q1-2018 CONFIRMÉ via ledger** (pas un artefact) : jambe indice +20 % de W, composants +2 %, **delta-hedge −154 % de W** = bleed de gamma sur clôtures SPX réelles (indépendant du modèle de marquage) — le canal « variance réalisée de l'indice explose quand la corrélation spike » (éq. 3 DMV). NAV intra-trimestre tombé à 6 % de W → sizing richesse inconditionnel = quasi-ruine (cohérent DMV Table VI marges)
- [✅] Grecques **relatives** (8 juil.) : `src/dispersion/utils/greeks.py` (prix/grecques BS + Black-76, conversions contrat↔relatif, forward par parité, IV implicite, straddles) + **10 tests pytest** (`tests/test_greeks.py`) : formes fermées, parité, différences finies, test anti-piège DMV (vega-neutralité en richesse), reproduction des primes SPX réelles du 28/06/2024 à ±0,5 %
- [✅] **Fork tranché (notebook 04, 8 juil.) : primes = `impl_premium`** (ré-extraction vsurfd — BS naïf biaisé de ±5 %, q implicite SPX 1,04 % identique call/put, couverture 100 % dès 1996) ; **grecques = BS/Black-76 analytiques** depuis IV + forward (`fwdprd`/parité call-put) + `zerocd` (README §8bis)
- [✅] **Fork tranché (8 juil.) : marquage QUOTIDIEN par interpolation de surface** (piliers < 91j, linéaire en variance totale à delta fixé) — delta-hedge quotidien effectif, P&L décomposable (vega/gamma/theta), séries quotidiennes pour le ML S4 ; holding-period DMV dispo par agrégation (benchmark) ; marquage à IV constante exclu (le P&L d'une stratégie de vol EST vega×ΔIV)
- [✅] **Rebuild groupé du dataset (8 juil., 2 passes ~10 min)** : `surface.parquet` (4,9 M lignes, piliers 10/30/60/91j, 0 NaN, 0 doublon) + `spots.parquet` (724k closes secprd + cfadj, 169/329 titres avec splits, SPX vérifié) + `rates.parquet` (zerocd) + `returns.parquet` (329 permnos dès 1995) + durcissements §9bis + fix des queues tronquées. Non-régression : 5 parquets bit-identiques ; signal modifié sur 139 jours (≤ 8e-3, politique documentée enfin appliquée) ; headline inchangé (+0.079, t=7.4). Découvertes QA : pilier 10j peuplé ~37 % (règle de repli sous 30j à poser au design engine) ; zerocd absent 10 j/7281 (ffill à la consommation).

- [✅] **Forks engine tranchés (8 juil., README §7.3)** : sizing = **richesse DMV** (comparabilité Table II — à justifier dans la thèse) ; delta-hedge = **indice seul quotidien** (cohérence one-factor + coûts ~0,5–2 bps vs 100 lignes actions — à justifier dans la thèse) ; échéance = **règlement intrinsèque** au rebal. suivant (±3 j documenté) ; marquage < 30j = **σ(30j) du jour** (règle homogène) ; approximation **ATM-proxy** documentée (primes réelles + payoff ⇒ second ordre sur le cumulé)

### Règles & réalisme
- [✅] **v1 EXÉCUTÉE (8 juil.)** — seuil = **quantile ex-ante glissant** (option A : médiane du signal sur l'historique disponible, warm-up 12 rebal., zéro look-ahead ; `exante_quantile_threshold`) : 58/115 trimestres tradés. Résultats : prime par trimestre TRADÉ améliorée (+7,9 % vs +7,3 % v0) → le signal a de l'info sur la prime ; **esquive Q3-2024 (+1,4 % vs −41 %), été 2002 et les trimestres Lehman** ; coupe la stratégie post-2020 (17/20 derniers trimestres sautés = la compression du notebook 03 en action) ; **MAIS trade le Q1-2018** (signal au-dessus de sa médiane fin 2017 : le spread mesure la richesse de la prime, PAS le danger) → maxDD inchangé, Sharpe global 0,56 < 0,77 (moitié du temps en cash). **Conclusion : le seuil time la prime, pas le risque — argument quantitatif exact pour le ML S4 (cible = régimes de spike, features de niveau de vol/spectrales).** Sensibilité q ∈ {0,25 ; 0,75} → robustesse S5. Sortie/roll : règlement intrinsèque (§7.3)
- [✅] **Fork coûts tranché (8 juil., notebook 05, README §8bis)** : grille **paramétrique calibrée sur vrais quotes opprcd** (3 groupes SPX / rnk 1-50 / rnk 51-100 × interpolation linéaire 1996→2024 : 1→0,6 % / 7→1,2 % / 10→3 %) ; **½-spread × prime par jambe à l'entrée** (spread payé une fois, convention DMV held-to-maturity) + **1 bp** sur le notionnel de chaque trade de hedge ; règlement sans spread. Raw opprcd rejeté : artefact EOD SPX 2012 (quoted ≠ effective, ~30-50 % du coté) + cellule COVID 2020. Conservateur par construction. Sensibilité ±50 % + « stress ×2 » → S5
- [✅] **NETS EXÉCUTÉS (8 juil.) — la Table V de DMV se réplique sur nos 29 ans** : v0 net = +4,0 %/trim (brut +7,3 %, ÷1,8), **Sharpe 0,77 → 0,42 (DMV : 0,73 → 0,41)**, cumul ×165 → ×2,5 (~3 %/an net, marginal) ; coût ~3,3 %/trimestre tradé ; **v1 net morte (×1,1, Sharpe 0,33)** — la porte sélectionne les époques à spreads larges (coût 3,0 % vs 2,7 %). Narratif limits-to-arbitrage complet + barre chiffrée pour le ML S4. Sorties : `backtest_{v0,v1}_net_*.parquet`

### Métriques
- [✅] Sharpe, Sortino, MaxDD, P&L cumulé, t-stats (notebook 06 → `results/tables/table_s3_metrics.csv`) : v0 brut t=4,11 ; **v0 net t=2,24 (encore significatif)** ; v1 net t=1,79 (marginal)
- [✅] **Décomposition du P&L par source** (ledger, additive) : tous trimestres = jambe indice +8,3 % / composants −2,4 % / hedge +1,2 % ; **crises (12/115) = hedge −23,0 %** (le canal variance réalisée/gamma) + indice −14,3 %, règlement +8,2 % — mappe sur (theta+vega = jambes, gamma/corrélation réalisée = hedge) ; granularité grecque fine = raffinement S5 si utile
- [✅] **Sanity DMV** (`table_s3_dmv_sanity.csv`) : Σy +99,5 % vs +101,12 % ✓ ; Sharpe 0,77/0,42 vs 0,73/0,41 ✓ ; skew négatif ✓ ; hedge d'entrée quasi delta-neutre (+0,3 %, |.| 5 %) — le −32,54 % DMV (1 mois, statique) non directement comparable, documenté
- [✅] Stats : t simples sur trimestres non chevauchants ; NW réservé aux séries quotidiennes (S5)

**Livrable :** ✅ `results/figures/fig_backtest_s3.png` (NAV log 4 variantes, crises annotées) + `results/tables/table_s3_{metrics,dmv_sanity}.csv` + notebook 06. **SEMAINE 3 TERMINÉE** (8 juil. — 11 jours d'avance sur le calendrier).

**Piège :** Delta-neutre ne suffit pas → surveiller le Vega-convexity (Volga).

---

## SEMAINE 4 (20 juil → 26 juil) — Random Matrix Theory (+ ML léger si le temps le permet)

**Objectif :** la couche de valeur ajoutée quantitative.

### RMT (prioritaire) — spec figée le 8 juil. (README §8bis), module `src/dispersion/rmt/`
- [ ] ⏸ Fork en suspens : ajouter `returns.parquet` au build (long : date, permno, ret — requis pour matrices 252j + features spectrales) ; schéma exact à trancher en début de S4
- [ ] Matrices de corrélation **252j** (q=N/T≈0,4 sain — les 63j stockées sont singulières, q≈1,6, spectre dégénéré) + spectre de valeurs propres
- [ ] Superposer Marchenko–Pastur ; λ₊ **effectif** avec correction Laloux : (1−λ₁/N)(1+√q)²
- [ ] Pipeline : dévolatilisation EWMA + standardisation → diagonalisation → clipping du bulk en λ̄ (**trace préservée**) → renormalisation (diag=1, PSD garantie)
- [ ] Tests unitaires : diag=1, PSD, trace + **test iid simulé** (histogramme ≈ densité MP ; filtre ≈ identité)
- [ ] Rôle A : ρ̄_realized débruitée → réinjecter dans le spread, re-backtester vs baseline
- [ ] Explorer la **jambe parcimonieuse** (structure spectrale → moins de noms → moins de spread payé = réponse RMT aux frictions DMV)
- [ ] Justifier les **deux fenêtres** dans le rapport (63j = window-matching de l'horizon pricé ; 252j = santé du spectre)
- [ ] Réfs : Bun–Bouchaud–Potters 2017 (arXiv 1610.08104), Potters–Bouchaud 2020

### ML (bonus, seulement si la RMT est bouclée)
- [ ] Features **spectrales** (rôle B de la RMT) : λ₁/N, absorption ratio, K (nb λ > λ₊), Δλ₁, rotation du vecteur propre dominant + VIX, vol réalisée, lags
- [ ] Cible : régimes de **spike de corrélation** → couper le short-corr (objectif : skew/drawdown, pas le rendement moyen = « n'entrer que si edge prédit > coût »)
- [ ] XGBoost + validation walk-forward avec purging/embargo (anti data-leakage)
- [ ] (Si vraiment du temps) K-Means pour régimes calme/stress/crise

**Livrable :** graphe du spectre + tableau comparatif baseline vs RMT (et vs ML si fait).

**Piège :** ne pas sur-vendre le RMT ; le data leakage est l'erreur n°1 si tu touches au ML.

---

## SEMAINE 5 (27 juil → 2 août) — Analyse, Robustesse & Rédaction

**Objectif :** transformer le code en rapport académique de 3000 mots.

- [ ] Stress-tests (COVID 2020, taux 2022) + sensibilité aux paramètres
- [ ] **Sous-périodes** : la prime s'est-elle comprimée post-2003 ? (prédiction falsifiable du framework limits-to-arbitrage — notre valeur ajoutée vs DMV : 29 ans vs leur échantillon arrêté début 2000s)
- [ ] Stats finales : **Newey–West (~63 lags)** partout où les fenêtres se chevauchent + tests non-chevauchants (trimestriels) en robustesse
- [ ] Analyse Vanna/Volga (au moins qualitative) + risque de convexité
- [ ] (Si temps) Robustesse MFIV : reconstruction Carr–Madan sur la jambe indice pour borner le biais de skew de l'ATM (README §5)
- [ ] Rédaction : Intro/Client Spec → Théorie → Données & Méthodo → Résultats → Limites → Conclusion
- [ ] **Narratif** : la prime existe (DMV + nos 29 ans) → les frictions la tuent (Tables V–VI) → RMT (jambe parcimonieuse) + ML (edge prédit > coût) = réponses aux frictions
- [ ] Soigner les figures + intégrer la reading list
- [ ] Nettoyer et documenter le repo

**Livrable :** rapport final + repo propre.

**Piège :** garde au moins 3-4 jours pleins pour rédiger, pas la veille.

---

## Vue d'ensemble du calendrier

| Semaine | Dates | Cœur |
|---|---|---|
| 1 | 29 juin → 5 juil | Setup + Données point-in-time |
| 2 | 6 → 12 juil | Théorie + Corrélation implicite |
| 3 | 13 → 19 juil | Backtest baseline |
| 4 | 20 → 26 juil | RMT (+ ML léger) |
| 5 | 27 juil → 2 août | Robustesse + Rédaction |

---

## Priorités défensives (si ça dérape)

1. **Semaines 1-3** (Données → Signal → Baseline) = projet complet et notable, à sécuriser coûte que coûte
2. **Semaine 4 RMT** = forte valeur ajoutée, rapide, à viser
3. **ML** = pur bonus ; ne le commence que si tout le reste tient. Mieux vaut un projet baseline+RMT impeccable qu'un ML bâclé.