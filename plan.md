# Plan de travail — ML-Driven Dispersion Trading Strategy

> **Cadrage calendaire :** démarrage 25 juin 2026, objectif de fin 31 juillet 2026 → ~5 semaines. Rythme intensif. Le ML est une couche bonus, pas un prérequis.

---

## SEMAINE 1 (25 juin → 2 juil) — Setup, Données & Univers

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

## SEMAINE 2 (3 juil → 9 juil) — Théorie & Signal

**Objectif :** la mécanique du signal + sa justification théorique.

### Dérivation mathématique
- [ ] P&L du portefeuille de dispersion via Lemme d'Itô
- [ ] Décomposition en pari corrélation réalisée vs implicite
- [ ] Formalisation du Volatility Risk Premium + Correlation Risk Premium

### Corrélation implicite
- [ ] Formule d'inversion : IV indice + composants + poids → ρ_implied
- [ ] ρ_realized depuis la covariance réalisée
- [ ] Série temporelle du **spread** (ρ_implied − ρ_realized) = signal brut
- [ ] Vérification : ρ_implied borné dans [0,1]

**Livrable :** série du dispersion spread + graphe montrant l'implicite en moyenne au-dessus du réalisé (validation empirique de la thèse).

**Piège :** approximation one-factor (corrélation moyenne unique) — à assumer et documenter.

---

## SEMAINE 3 (10 juil → 16 juil) — Stratégie Baseline

**Objectif :** un backtest qui tourne, sans ML = benchmark de référence.

### Moteur
- [ ] Classe `DispersionEngine` (ouverture/fermeture, suivi P&L)
- [ ] Portefeuille : long straddles composants, short straddle indice
- [ ] Greeks 1er ordre (Delta, Vega) ; Vega-neutre à l'entrée, Delta-neutre en rééquilibrage quotidien

### Règles & réalisme
- [ ] Règle d'entrée (seuil de spread), sortie/roll, gestion du roll à l'échéance
- [ ] Coûts de transaction (bid-ask options), dividendes, slippage

### Métriques
- [ ] Sharpe, Sortino, Max Drawdown, P&L cumulé
- [ ] Décomposition du P&L par source (vega, gamma, theta, correlation)

**Livrable :** courbe de P&L cumulé baseline + tableau de métriques. **C'est le cœur notable du projet.**

**Piège :** Delta-neutre ne suffit pas → surveiller le Vega-convexity (Volga).

---

## SEMAINE 4 (17 juil → 23 juil) — Random Matrix Theory (+ ML léger si le temps le permet)

**Objectif :** la couche de valeur ajoutée quantitative.

### RMT (prioritaire)
- [ ] Matrice de corrélation réalisée N×N + spectre de valeurs propres
- [ ] Superposer la densité de Marchenko-Pastur (bulk de bruit vs signal)
- [ ] Eigenvalue clipping → matrice "propre"
- [ ] Réinjecter dans le spread/les poids, re-backtester, comparer baseline vs RMT

### ML (bonus, seulement si la RMT est bouclée)
- [ ] XGBoost : prédire le spread à H jours + feature importance (VIX, vol réalisée, lags)
- [ ] Validation walk-forward avec purging/embargo (anti data-leakage)
- [ ] (Si vraiment du temps) K-Means pour régimes calme/stress/crise

**Livrable :** graphe du spectre + tableau comparatif baseline vs RMT (et vs ML si fait).

**Piège :** ne pas sur-vendre le RMT ; le data leakage est l'erreur n°1 si tu touches au ML.

---

## SEMAINE 5 (24 juil → 31 juil) — Analyse, Robustesse & Rédaction

**Objectif :** transformer le code en rapport académique de 3000 mots.

- [ ] Stress-tests (COVID 2020, taux 2022) + sensibilité aux paramètres
- [ ] Analyse Vanna/Volga (au moins qualitative) + risque de convexité
- [ ] Rédaction : Intro/Client Spec → Théorie → Données & Méthodo → Résultats → Limites → Conclusion
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