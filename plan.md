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
- [✅] Formule d'inversion implémentée + validée (notebook 03, 8 juil.) — reste : promotion `src/dispersion/signal/implied_corr.py` + écriture `signal.parquet`
- [✅] ρ_realized depuis la covariance réalisée (fait en S1 : ρ̄ pondérée quotidienne, option B — même forme fonctionnelle que l'inversion ⟹ spread pomme-à-pomme)
- [✅] Séries **premium** et **signal** calculées : Π̄ = **+0.079 (t-NW63 = 7.4, 75 % j > 0)**, S̄ = +0.078 (t = 10.6, 80 % j > 0) → **thèse CRP validée sur 1996–2024**
- [✅] Vérification bornes : ρ_implied ∈ [0.12, 0.91] — **0 violation, 0 NaN** sur 7221 j ; couverture min 91/100 (plancher jamais atteint)
- [✅] Aperçu sous-périodes : 2008–12 = 0.135 (t=4.3) ; **2020–24 = 0.028 (t=1.19, non significatif)** → compression récente de la prime, à creuser en S5 (limits-to-arbitrage)
- [✅] Promotion : `src/dispersion/signal/implied_corr.py` (`implied_correlation` + `build_signal`) + `signal.parquet` écrit (7221 j × 7 col.) — reproduit le notebook à l'identique. **SEMAINE 2 TERMINÉE** (rédaction des dérivations → S5).

**Livrable :** ✅ `signal.parquet` + graphe de validation `results/fig_crp_validation.png` (implicite au-dessus du réalisé forward, premium window-matché).

**Piège :** approximation one-factor (corrélation moyenne unique) — à assumer et documenter (DMV éq. 6 fait la même hypothèse).

---

## SEMAINE 3 (13 juil → 19 juil) — Stratégie Baseline

**Objectif :** un backtest qui tourne, sans ML = benchmark de référence.

### Moteur
- [ ] Classe `DispersionEngine` (ouverture/fermeture, suivi P&L)
- [ ] Portefeuille : long straddles composants, short straddle indice
- [ ] **v0 inconditionnel D'ABORD** (trade systématique à chaque rebal. — seul comparable aux benchmarks DMV Table II), puis v1 conditionné au seuil de signal — sépare « la prime existe » de « on sait la timer »
- [ ] Greeks 1er ordre (Delta, Vega) ; Vega-neutre à l'entrée, Delta-neutre en rééquilibrage quotidien — hedge triangulaire DMV éq. (10)–(11) : vega d'abord, delta résiduel avec l'indice
- [ ] Grecques **relatives** (vega/prix, par dollar investi) — conversions brut↔relatif dans `utils` + **test unitaire** sur cas BS en forme fermée (piège : mélange de conventions = hedge silencieusement faux)
- [ ] ⏸ Fork en suspens : source des primes/grecques — `impl_premium` (vsurfd, interpolation entre maturités pour marquer les positions vieillissantes) vs recalcul Black–Scholes (+ `zerocd`, dividendes). À trancher après exploration de la surface.

### Règles & réalisme
- [ ] Règle d'entrée (seuil de spread) pour v1, sortie/roll, gestion du roll à l'échéance
- [ ] Coûts de transaction — leçon DMV Table V : **1er ordre** (rendement ÷2, alpha mort). Convention : bid-to-maturity (jambes vendues) / ask-to-maturity (jambes achetées), spread payé une fois.
- [ ] ⏸ Fork en suspens : vrais bid/ask `opprcd` (lourd, matching 91j↔contrats listés) vs spread paramétrique calibré sur échantillon `opprcd` + sensibilité

### Métriques
- [ ] Sharpe, Sortino, Max Drawdown, P&L cumulé
- [ ] Décomposition du P&L par source (vega, gamma, theta, correlation)
- [ ] **Sanity vs benchmarks DMV** (README §8bis) : poids ~−100/+101/−32,5, ratio 0,58, Sharpe ~0,73 brut / 0,41 net, β≈0 — écart massif = chasse au bug avant interprétation
- [ ] Stats : **Newey–West (~63 lags)** sur toute série chevauchante

**Livrable :** courbe de P&L cumulé v0 (benchmark DMV) + v1 (signal) + tableau de métriques. **C'est le cœur notable du projet.**

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
| 1 | 25 juin → 2 juil | Setup + Données point-in-time |
| 2 | 3 → 9 juil | Théorie + Corrélation implicite |
| 3 | 10 → 16 juil | Backtest baseline |
| 4 | 17 → 23 juil | RMT (+ ML léger) |
| 5 | 24 → 31 juil | Robustesse + Rédaction |

---

## Priorités défensives (si ça dérape)

1. **Semaines 1-3** (Données → Signal → Baseline) = projet complet et notable, à sécuriser coûte que coûte
2. **Semaine 4 RMT** = forte valeur ajoutée, rapide, à viser
3. **ML** = pur bonus ; ne le commence que si tout le reste tient. Mieux vaut un projet baseline+RMT impeccable qu'un ML bâclé.