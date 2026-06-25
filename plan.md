# Plan de travail — ML-Driven Dispersion Trading Strategy

> **Cadrage calendaire :** démarrage 25 juin 2026, objectif de fin 31 juillet 2026 → ~5 semaines. Rythme intensif. Le ML est une couche bonus, pas un prérequis.

---

## SEMAINE 1 (25 juin → 2 juil) — Setup, Données & Univers

**Objectif :** infrastructure prête + dataset propre point-in-time. C'est la semaine la plus critique.

### Setup
- [ ] Connexion WRDS opérationnelle (`wrds.Connection()`) — relancer l'admin si toujours "under review"
- [ ] Vérifier l'accès à `optionm.vsurfd` et aux tables CRSP de constituents
- [ ] Repo GitHub + environnement (`requirements.txt` : wrds, pandas, numpy, scipy, scikit-learn, statsmodels)
- [ ] Figer les **paramètres de design** : N composants (30 conseillé pour aller vite), maturité 91j, delta 50 (ATM), période de backtest (2018-2024 suffit), rebalancement trimestriel

### Données
- [ ] CRSP : constituents S&P 500 par rebalancement (memberships + poids) → fonction `get_universe(date)`
- [ ] OptionMetrics `vsurfd` : IV pour SPX + composants, maturité 91j, delta 50
- [ ] Jointures d'identifiants : `secid` ↔ `permno` ↔ ticker
- [ ] CRSP `dsf` : prix ajustés → rendements → vol réalisée glissante (21j, 63j) + corrélation réalisée
- [ ] Nettoyage : trous, outliers IV, alignement des calendriers

**Livrable :** fichiers `.parquet` propres (`iv_index`, `iv_components`, `weights`, `realized_vol`, `realized_corr`) alignés.

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