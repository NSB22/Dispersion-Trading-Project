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
- [✅] `returns.parquet` : fait en S3 (rebuild groupé anticipé — 329 permnos, buffer 1995)
- [✅] **Forks étape 0 tranchés (17 juil.)** : notation `q_mp` = N/T (code + note README ; `q` reste le dividende dans le pricing) ; dévol **EWMA λ=0,94** (RiskMetrics, sensibilité S5) ; matrices nettoyées **quotidiennes** ; réinjection = **les deux voies** (variante signal_rmt complète re-backtestée + features spectrales pour le ML) ; Laloux = partie intégrante du pipeline (fixe le bord de clipping), pas d'estimateur séparé
- [✅] **Étape 1 (17 juil., notebook 07)** : matrices 252j dévolatilisées (EWMA 0,94) aux 116 rebalancements — q_mp médian 0,397, N_eff ≈ 100 ; **λ₁/N médiane 33,5 %** [17 % fin 2017 → 54 % mi-2012] ; spectre vs MP : le bulk colle à la densité **avec σ² = 1−λ₁/N** (Laloux) ; **K médian = 7 au bord Laloux vs 4 au bord naïf** (le bord naïf, trop haut de ~35 %, RATE ~3 facteurs sectoriels/trimestre — démonstration chiffrée de la correction) ; **test iid : K = 0**, bords exacts, trace 1.0000 (le pipeline n'invente rien) ; aperçu clipping : ρ̄_clean vs ρ̄_raw corr 0,998, |Δ| médiane 0,011 (max 0,024) → **le scalaire bouge peu comme anticipé** (λ₁ préservé) — l'effort RMT portera sur features + jambe parcimonieuse. Figure : `fig_mp_spectrum.png`
- [✅] **Étape 2 (17 juil.) — module `src/dispersion/rmt/cleaning.py`** : `devolatilise` (EWMA 0,94, σ prévisible shift-1), `corr_window` (252j complete-case, min 200 obs/60 noms), `laloux_clip` (bord effectif → clipping trace-preserving → renorm diag=1) avec diagnostics (q_mp, λ₁/N, K, bords, témoin de trace), `spectral_features` (λ₁/N, K, absorption top-5, v1 signé pour la rotation)
- [✅] **Tests (6, suite 31/31)** : iid → K=0 + quasi-identité (off-diag ÷20+) ; diag/PSD/trace sur données corrélées ; **équicorrélation exacte → le nettoyage ne déforme PAS la vraie structure** (diff < 1e-10) ; invariance d'échelle en log ; panel trop mince → None ; cohérence features/clip
- [✅] **Étape 3 — Rôle A EXÉCUTÉ (17 juil.)** : `rmt/daily.py` (`build_rmt_daily` 61 s → `rmt_daily.parquet` 7221 j × [ρ̄ raw/clean 252j, λ₁/N, K, absorption, Δλ₁, rotation] ; `build_signal_rmt` → `signal_rmt.parquet`) ; corr(signal_base, signal_rmt) = 0,40. **v1_rmt (porte médiane ex-ante sur signal_rmt) : Sharpe brut 0,80 (v1 : 0,56), skew +1,28 (POSITIF vs −1,45), maxDD −51,7 % (vs −95,9 %), esquive LES TROIS crashs dont le Volmageddon ; net : Sharpe 0,56 > barre 0,42, skew +0,80, cumul ×12,4.** 39/115 désaccords de porte vs v1.
- [✅] **ABLATION D'ATTRIBUTION (le garde-fou « ne pas sur-vendre la RMT »)** : la même porte sur la trailing 252j BRUTE fait 0,78 / skew +1,31 / mêmes crashs esquivés (2 désaccords/115 vs nettoyée) → **le moteur est la FENÊTRE 252j (ancre lente de régime), le clipping n'ajoute que ~+0,02 de Sharpe sur ce canal**. La valeur RMT se joue sur les features (rôle B/ML) et la jambe parcimonieuse — à écrire tel quel dans la thèse. Intégrité : 0 skip forcé par NaN aux rebalancements (85 j NaN = churn mi-trimestre 2001/2009/2016).
- [✅] **Audit adversarial look-ahead du chemin RMT (agent indépendant, 17 juil.) : AUCUNE fuite** — EWMA prévisible (invariance bit-à-bit aux perturbations futures), fenêtres bornées à t (convention identique à la baseline), shift(−63) exact, porte reconstruite = 115/115 décisions du run, quantile ex-ante inclusif = 0/115 décision changée. **2 défauts trouvés et CORRIGÉS** : ① colonne `premium` de signal_rmt **invalidée** (trailing 252j vs son shift-63 = 75 % d'observations partagées, corr signal/premium 0,876 — contamination mécanique, ne JAMAIS l'utiliser en validation ; le premium de thèse reste le 63j window-matché de la base) ; ② **bord MP à T nominal au lieu de T effectif** (complete-case T_eff ∈ [200,252] sur ~⅓ des jours → k_signal gonflé de ~1 ; du bruit iid à T_eff=201 passé comme 252 montrait K=2) → fix : `corr_window` retourne T_eff, bord calculé dessus + test de régression ; + fallback silencieux des vols remplacé par échec bruyant. Suite : 34/34. Consigné : mélange d'estimateurs (corr dévolatilisée × vols brutes — variante assumée, à écrire), features rotation/Δλ₁ trouées les 85 j de churn (à documenter pour le ML).
- [ ] Explorer la **jambe parcimonieuse** (structure spectrale → moins de noms → moins de spread payé = réponse RMT aux frictions DMV)
- [ ] Justifier les **deux fenêtres** dans le rapport (63j = window-matching de l'horizon pricé ; 252j = santé du spectre)
- [ ] Réfs : Bun–Bouchaud–Potters 2017 (arXiv 1610.08104), Potters–Bouchaud 2020

### ML (enrichi le 17 juil. — on a le temps : régimes via HMM/GMM en plus du supervisé)
- [ ] Features **spectrales** (rôle B de la RMT) : λ₁/N, absorption ratio, K (nb λ > λ₊^eff), Δλ₁, rotation du vecteur propre dominant + VIX/vol réalisée, lags
- [ ] Cible : régimes de **spike de corrélation** → couper le short-corr (objectif : skew/drawdown, pas le rendement moyen = « n'entrer que si edge prédit > coût »)
- [✅] **GMM + HMM régime = feature de stacking (17 juil., `ml/regime.py`)** : GMM(3) pointwise + HMM(3) **filtré forward MANUEL** (jamais Baum-Welch lissé), refit walk-forward, état danger = centroïde de stress composite (LABEL-FREE → 0 look-ahead), standardisation train-only. **3 tests de causalité** (perturbation du futur → passé bit-identique). Éco validée : crises → 1,00 (dont **Volmageddon 0,98, que le spread ratait**), calme → 0,00–0,07. Bug corrigé : turbulence seule aplatie par la dévol → centroïde composite.
- [✅] **Étapes ④-⑤ EXÉCUTÉES (17 juil., `ml/metamodel.py`, notebook 09)** : méta-modèle XGBoost peu profond (fixe, non tuné sur 12 crises), walk-forward purge 63j + embargo 21j, prédit le rendement du trade, veto au quantile ex-ante 33e ; test central VIX / VIX+spectral / Full.
  **RÉSULTAT = NUL (honnête)** : pouvoir prédictif quasi inexistant (corr ŷ,y : VIX +0,02, **spectral −0,14, Full −0,20**) → le rendement trimestriel du trade est imprévisible à cet échantillon. Le veto ML n'améliore PAS v1_rmt (net 0,56 → 0,57, bruit ; VIX-seul *dégrade* 0,51). **Les features spectrales ne battent pas le VIX** (VIX-seul a même meilleur recall crises 38 % vs 0 %). **Découverte clé : détecter un régime de stress ≠ prédire que le trade perdra** (le stress est souvent quand la prime est la plus riche). Respecte le piège du plan « ne pas sur-vendre le ML ».
- [✅] **La vraie valeur S4 = la fenêtre RMT 252j** (v1_rmt, étape 3), pas le ML : Sharpe net 0,42→0,56, skew −1,33→+0,80, maxDD −99 %→−58,5 %, cumul ×2,5→×12,4.
- [✅] **Menu d'améliorations ML présenté + tranché (17 juil.)** : architecture = **méta-labeling veto** (v1_rmt propose, le ML veto — *choix NON définitif, à réévaluer* ; porte directe en comparaison si temps) ; retenus dès S4 : **labels continus** (spike Δρ̄ forward), **porte coût-aware** (edge prédit > coût d'époque), **ensembling GMM/HMM/XGB**, **calibration isotonique** ; reportés S5 : CPCV + Deflated Sharpe/PBO.
- [✅] **PROTOCOLE PRÉ-ENREGISTRÉ (figé le 17 juil. AVANT tout résultat — anti multiple-testing)** :
  - **Features — PROTOCOLE v2 (enrichi le 17 juil. AVANT tout résultat walk-forward, demande utilisateur)** : 32 features en 5 blocs — **A. spectral profond** (λ₁/N, λ₂/N, K, absorption top-5, entropie spectrale, participation ratio v1, dispersion cross-section des corrélations, Δλ₁ 21j, rotation 21j, **turbulence de Mahalanobis sur l'inverse NETTOYÉE** brute + 21j) ; **B. structure de vol** (IV SPX + Δ21j, pente de terme 30→91j, skew ±50δ, vol-of-vol) ; **C. réalisé & VRP** (RV21/63 SPX, **VRP = IV−RV63**, drawdown 252j, momentum 63j) ; **D. cross-section IV** (IV composantes pondérée, dispersion des IV) ; **E. corrélation/signal** (ρ_imp, trail63, clean252, Δρ 21j, signaux base+rmt, écart d'ancres, percentile ex-ante expanding du signal, coût d'époque). **Discipline dimensionnelle** : les 32 pour XGBoost ; core set FIGÉ de 8 pour GMM/HMM (f_lam1, f_dlam1_21, f_turb21, f_vrp, f_term_slope, f_anchor_gap, f_iv_spx, f_rot21). `ml_dataset.parquet` = 7221 × 34.
  - **Label continu** : y(t) = ρ̄_trail63(t+63) − ρ̄_trail63(t) — le spike de corrélation réalisé sur le trimestre suivant (usage ENTRAÎNEMENT uniquement, purgé).
  - **Modèles/grilles** : GMM 3 composantes full-cov ; HMM gaussien 3 états (probas FILTRÉES) ; XGBoost régression {depth 2-3 ; n_est 100/300 ; lr 0.05/0.1 ; subsample 0.8} sélection par CV purgée INTERNE à la fenêtre d'entraînement.
  - **Validation** : walk-forward expanding refit à chaque rebalancement, purge 63 j ouvrés + embargo 21 j ; seuil de veto = quantile ex-ante des prédictions (médiane, symétrie avec v1).
  - **Métriques figées** : tableau v0/v1/v1_rmt/ML × brut/net — Sharpe, skew, maxDD, trades, crashs évités ; **barre = Sharpe net 0,42**. Discipline : aucune stat prédictive full-sample regardée avant l'évaluation walk-forward.
- [🔄] **PROTOCOLE v3 (17 juil., directive utilisateur) — « calculer large, sélectionner étroit »** : le modèle final ne verra que **5–8 features** (jamais les ~35 en direct ; HMM/GMM : 3–4 max). **Pipeline gated, arrêt de validation à CHAQUE étape** : ① audit point-in-time (tableau feature|fenêtre|source|PIT) → ② dérédondance (clusters corr > 0,8, garder la plus interprétable, documenter les éliminées) → ③ [DISCUSSION LABEL — **réouverte**, y_spike = candidate non actée] → ③bis génération des **probas de régime GMM/HMM comme features candidates** (STACKING, tranché 17 juil. — fit non supervisé sans label, étiquetage danger sur historique d'entraînement uniquement, walk-forward anti-fuite ; sur core set 3–4) → ④ sélection par importance XGBoost walk-forward purgé → ⑤ **TEST CENTRAL : « VIX seul » (f_vix + term structure) vs « VIX + spectrales retenues »**
- [✅] **Étape ② dérédondance FAITE (17 juil., notebook 08) — validée par l'utilisateur** : 35→26 (9 éliminées, 3 clusters |ρ|>0,8, survivant le plus interprétable ; stabilité 2-moitiés min 0,68 ; 0 look-ahead car feature-feature). Puis **resserrage à 10 par jugement économique** (directive utilisateur « garde 10 et avance » ; label-free = plus robuste que l'importance in-sample sur 12 crises) : **f_vix, f_term_slope** (baseline vol) + **f_lam1, f_dlam1_63, f_turb21, f_k, f_rot21** (spectral RMT) + **f_vrp, f_drho_imp_21, f_sig_rmt** (éco/signal). Core HMM/GMM = {f_lam1, f_dlam1_63, f_turb21, f_vix}. **Proba régime ajoutée PAR-DESSUS (11e input, quoi qu'il arrive — directive utilisateur, pas de sélection pour elle).** f_cost_era → porte coût-aware uniquement. `ml_features_final.json`. — les features RMT apportent-elles du pouvoir prédictif AU-DELÀ du VIX ? (le résultat à soigner, plus important que le Sharpe). Métriques : prédictives (précision/rappel crises) ET économiques (Sharpe net), incertitude explicite (12 crises). **Ajouts** : `f_vix` (CBOE `cboe.cboe`, 1996–2024, 2 doublons vendeur dédupliqués ; **VIX3M INDISPONIBLE dans l'abonnement** → term structure = piliers ATM 30/91j vsurfd = `f_term_slope`, full-sample, convention pente documentée) ; `f_drho_imp_21` (le marché commence à pricer) ; `f_dlam1_63` (spec HMM) ; proxy liquidité HY **skippé** (non sourçable proprement). `ml_dataset.parquet` = 7221 × 37 (35 features + date + y_spike-candidate).

**Livrable :** ✅ `fig_mp_spectrum` + `fig_ml_nav` + `fig_central_test` + `fig_regime_timeline` + `fig_feature_importance` + `fig_pred_scatter` + `fig_return_dist` + `table_ml_metrics/central_test/feature_importance`. **SEMAINE 4 = TERMINÉE (17 juil.)** : RMT = vraie valeur (fenêtre 252j, Sharpe net 0,56, queues coupées) ; ML = résultat nul honnête (pas de pouvoir prédictif du rendement du trade, spectral ne bat pas VIX). Audit look-ahead RMT passé ; 3 tests de causalité HMM.

**Piège :** ✅ respecté — RMT non sur-vendu (ablation : c'est la fenêtre, pas le clipping) ; ML non sur-vendu (nul assumé) ; 0 data leakage (audit adversarial + tests de causalité).

- [✅] **Leviers d'amélioration 1+3 (17 juil., `ml/experiments.py`, en parallèle)** : **(1) cible = spike quotidien** (7000 obs) → **corr(ŝ, spike) OOS = +0,40** (vs rendement +0,02/−0,20) : le spike EST prévisible → la cible était le problème. MAIS le veto spike n'améliore pas (Sharpe net 0,53) — le danger est pricé (confirmation stress≠perte). **(3) régime seul comme porte** → modeste gain de queue : skew net +1,00 (vs +0,80), maxDD −52,4 % (vs −58,5 %), Sharpe 0,57 ; brut skew +1,51, maxDD −44,8 %. Seul levier qui améliore le profil de risque. **→ À adopter comme variante en S5.**

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