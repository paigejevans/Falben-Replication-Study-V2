"""
HDDM Analysis — Falbén et al. (2020) Replication
"It's not always about me: The effects of prior beliefs and stimulus
 prevalence on self-other prioritisation"
QJEP 73(9), 1466-1480. DOI: 10.1177/1747021820913016

WHAT THIS SCRIPT DOES (in order):
  1. Preprocesses all 6 CSVs → HDDM-ready format
  2. Fits 12 HDDM models (best + runner-up × 3 exps × 2 replications)
  3. Compares DIC values → confirms best model matches paper
  4. Extracts parameters (mean, 2.5q, 97.5q) → matches paper Tables 2,4,6
  5. Computes Bayesian p-values for z-bias → primary test in paper
  6. Runs Posterior Predictive Checks → verifies model fit
  7. Produces final summary table: successful vs failed side-by-side

MODELS FITTED (per paper Tables 1, 3, 5):
  Exp1: Model 7 (best) + Model 5 (runner-up)
  Exp2: Model 7 (best) + Model 2 (runner-up)
  Exp3: Model 4 (best) + Model 5 (runner-up)

MCMC: 10,000 samples + 1,000 burn-in (matches paper p.1471)

FIX NOTE:
  HDDM's depends_on does not accept lists. Where the paper specifies
  v varying by both expectancy and owner, we create a combined column
  "stim_cond" = expectancy + "_" + owner (e.g. "none_self", "equal_friend").
  This is equivalent and produces identical model structure.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

INPUT_DIR  = "/Users/paigeevans/Desktop/Replication Data Analysis"
OUTPUT_DIR = "/Users/paigeevans/Desktop/Replication Data Analysis/hddm_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = 10_000
BURN    = 1_000

# Models to fit per experiment — (name, depends_on dict)
# Chosen as best + runner-up from paper DIC tables.
# v uses "stim_cond" (= expectancy + "_" + owner) instead of a list,
# which is equivalent but compatible with HDDM's depends_on parser.
EXP_MODELS = {
    1: [
        ("Model_7", {"v": "stim_cond",
                     "a": "expectancy", "z": "expectancy"}),
        ("Model_5", {"v": "stim_cond",
                     "z": "expectancy"}),
    ],
    2: [
        ("Model_7", {"v": "stim_cond",
                     "a": "expectancy", "z": "expectancy"}),
        ("Model_2", {"z": "expectancy"}),
    ],
    3: [
        ("Model_4", {"a": "expectancy", "z": "expectancy"}),
        ("Model_5", {"v": "stim_cond",
                     "z": "expectancy"}),
    ],
}

# Paper's reported best model per experiment for verification
PAPER_BEST = {1: "Model_7", 2: "Model_7", 3: "Model_4"}

# Expectancy levels per experiment
EXP_EXPECTANCY = {
    1: ["none", "equal"],
    2: ["self", "friend", "equal"],
    3: ["self", "friend"],
}


# ═══════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESSING
# WHY:  Raw CSVs use column names and units HDDM cannot accept.
#       Five transformations required, verified against paper.
# NEXT: Preprocessed dataframes feed directly into HDDM fitting.
# ═══════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, exp_num: int) -> pd.DataFrame:
    """
    Transform raw CSV → HDDM-ready dataframe.

    Input:  participant, expectancy, owner, rt (ms), accuracy
    Output: subj_idx (int), rt (s), response (0/1), expectancy, owner,
            condition (expectancy + "_" + owner)

    1. participant → subj_idx (int) — required by HDDM
    2. rt / 1000 — HDDM requires seconds; wrong units corrupt estimates
    3. Filter rt < 0.2s — paper's 200ms exclusion (pp.1470-1474)
       No upper cutoff — response window was 2000ms (paper p.1470)
    4. Reconstruct response — paper: upper threshold = self-response
       (owner==self, correct) OR (owner==friend, incorrect) → response=1
       (owner==self, incorrect) OR (owner==friend, correct) → response=0
    5. Build condition column — expectancy + "_" + owner, required
       because HDDM depends_on does not accept lists
    """
    df = df.copy()

    # 1. participant → subj_idx
    df = df.rename(columns={"participant": "subj_idx"})
    if df["subj_idx"].dtype == object:
        labels  = sorted(df["subj_idx"].unique())
        mapping = {lbl: i for i, lbl in enumerate(labels)}
        df["subj_idx"] = df["subj_idx"].map(mapping)
    df["subj_idx"] = df["subj_idx"].astype(int)

    # 2. Milliseconds → seconds
    df["rt"] = df["rt"] / 1000.0

    # 3. Drop missing, then filter fast trials
    n_before = len(df)
    df = df.dropna(subset=["rt", "accuracy"])
    df = df[df["rt"] >= 0.2].copy()
    n_removed = n_before - len(df)
    pct = 100 * n_removed / n_before
    print(f"    Removed {n_removed} fast/missing trials ({pct:.2f}%)")

    # 4. Reconstruct response column
    df["response"] = (
        (df["owner"] == "self") == (df["accuracy"] == 1)
    ).astype(int)

    # 5. Combined condition column for v depends_on
    df["stim_cond"] = df["expectancy"] + "_" + df["owner"]

    # 6. Force all categorical columns to numpy object dtype.
    #    Pandas may read CSVs as StringDtype (extension type) which
    #    kabuki/HDDM's column validator does not recognise, silently
    #    causing the depends_on AssertionError.
    for col in ["expectancy", "owner", "stim_cond"]:
        df[col] = df[col].astype(object)

    # Assertions
    assert df["response"].isin([0, 1]).all()
    assert (df["rt"] >= 0.2).all()
    assert df["rt"].max() < 5.0, f"Unexpected max RT: {df['rt'].max():.2f}s"

    self_ok   = ((df[df.owner == "self"]["response"] == 1) ==
                 (df[df.owner == "self"]["accuracy"] == 1)).all()
    friend_ok = ((df[df.owner == "friend"]["response"] == 0) ==
                 (df[df.owner == "friend"]["accuracy"] == 1)).all()
    assert self_ok and friend_ok, "Response coding inconsistency"

    print(f"    Clean trials: {len(df):,}  |  "
          f"participants: {df['subj_idx'].nunique()}  |  "
          f"RT: {df['rt'].min():.3f}–{df['rt'].max():.3f}s")
    print(f"    Conditions: {sorted(df['stim_cond'].unique())}")

    return df[["subj_idx", "rt", "response",
               "expectancy", "owner", "stim_cond"]].copy()


# ═══════════════════════════════════════════════════════════════
# STEP 2 — HDDM MODEL FITTING
# WHY:  Decomposes RT + accuracy into v, a, z, t0 via hierarchical
#       Bayesian estimation. Fitting best + runner-up lets us verify
#       DIC ordering matches paper before trusting parameters.
# NEXT: Steps 3-7 all computed from these fitted model objects.
# ═══════════════════════════════════════════════════════════════

def fit_model(data, model_name, depends_on, exp_label, rep_label):
    """
    Fit one HDDM model.
    include=["sv","sz","st"] matches paper p.1471.
    response coded: upper=self, lower=friend — matches paper p.1470.
    """
    import hddm

    print(f"\n    Fitting {exp_label} {rep_label} — {model_name}")
    print(f"    depends_on: {depends_on}")

    db_name = os.path.join(
        OUTPUT_DIR,
        f"{exp_label}_{rep_label}_{model_name}.db"
    )

    # include=['z'] is required: z is fixed at 0.5 by default and is not
    # created as a free node unless explicitly included. Without this,
    # depends_on for z causes kabuki's post-build assertion to fail because
    # no z nodes are created and kabuki cannot find z's dependency column
    # in nodes_db. sv/sz/st are NOT included — those trigger full_ddm mode
    # which conflicts with multi-parameter depends_on.
    # Every parameter named in depends_on must appear in include,
    # otherwise kabuki's post-build node check throws an AssertionError.
    # We always add 't' as well (non-decision time, standard to estimate).
    include = list(set(list(depends_on.keys()) + ["t"]))
    m = hddm.HDDM(
        data,
        depends_on=depends_on,
        include=include,
    )
    m.find_starting_values()
    m.sample(
        SAMPLES,
        burn=BURN,
        dbname=db_name,
        db="pickle",
        progress_bar=True,
    )
    m.save(os.path.join(
        OUTPUT_DIR, f"{exp_label}_{rep_label}_{model_name}"
    ))
    print(f"    → DIC = {m.dic:.3f}")
    return m


# ═══════════════════════════════════════════════════════════════
# STEP 3 — DIC COMPARISON
# WHY:  Lower DIC = better fit (penalises model complexity).
#       Verifies our best model matches paper Tables 1, 3, 5.
# NEXT: Best model is used for Steps 4-6.
# ═══════════════════════════════════════════════════════════════

def compare_dic(models_dict, exp_label, rep_label, exp_num):
    rows   = [(name, m.dic) for name, m in models_dict.items()]
    dic_df = (pd.DataFrame(rows, columns=["Model", "DIC"])
              .sort_values("DIC")
              .reset_index(drop=True))

    best  = dic_df.iloc[0]["Model"]
    match = ("✓ matches paper"
             if best == PAPER_BEST[exp_num]
             else "⚠ differs from paper")

    print(f"\n    ── DIC: {exp_label} {rep_label} ──")
    print(dic_df.to_string(index=False))
    print(f"    Best: {best}  {match}")

    dic_df.to_csv(
        os.path.join(OUTPUT_DIR,
                     f"{exp_label}_{rep_label}_DIC.csv"),
        index=False
    )
    return best, dic_df


# ═══════════════════════════════════════════════════════════════
# STEP 4 — PARAMETER EXTRACTION
# WHY:  Posterior means + credible intervals are the values we
#       compare to paper Tables 2, 4, 6.
# NEXT: z parameters feed into Step 5 bias tests.
# ═══════════════════════════════════════════════════════════════

def extract_parameters(model, exp_label, rep_label, model_name):
    stats  = model.gen_stats()
    params = stats[["mean", "2.5q", "97.5q"]].copy()
    params.columns = ["Mean", "2.5q", "97.5q"]

    print(f"\n    ── Parameters: {exp_label} {rep_label} ──")
    print(params.round(3).to_string())

    params.to_csv(
        os.path.join(OUTPUT_DIR,
                     f"{exp_label}_{rep_label}_{model_name}_params.csv")
    )
    return params


# ═══════════════════════════════════════════════════════════════
# STEP 5 — BAYESIAN P-VALUES FOR Z BIAS
# WHY:  This is the primary test in the paper. z deviating from
#       0.50 indicates a response bias — the core finding.
#       pBayes < .05 = "extremely strong evidence" (paper p.1471)
# NEXT: Results feed into the Step 7 summary comparison.
# ═══════════════════════════════════════════════════════════════

def test_z_bias(model, exp_num, exp_label, rep_label, model_name):
    """
    For each expectancy condition, compute:
      pBayes(z > 0.50) = proportion of posterior supporting self-bias
      pBayes(z < 0.50) = proportion of posterior supporting friend-bias
    """
    conditions = EXP_EXPECTANCY[exp_num]
    rows = []

    for cond in conditions:
        node = None
        for name_try in [f"z({cond})", f"z_({cond})",
                         f"z.{cond}", "z"]:
            try:
                node = model.nodes_db.loc[
                    model.nodes_db.index.str.startswith(name_try)
                ].iloc[0]
                break
            except (KeyError, IndexError):
                continue

        if node is None:
            print(f"    ⚠ Could not find z node for: {cond}")
            continue

        try:
            trace = node["node"].trace()
        except Exception:
            print(f"    ⚠ Could not extract trace for z({cond})")
            continue

        mean_z   = float(np.mean(trace))
        p_self   = float(np.mean(trace > 0.50))
        p_friend = float(np.mean(trace < 0.50))
        p_bias   = min(p_self, p_friend)
        direction = "self" if mean_z > 0.50 else "friend"
        evidence  = (
            "extremely strong" if p_bias < .05  else
            "strong"           if p_bias < .10  else
            "moderate"         if p_bias < .20  else
            "none"
        )

        rows.append({
            "experiment":    exp_label,
            "replication":   rep_label,
            "stim_cond":     cond,
            "mean_z":        round(mean_z, 3),
            "pBayes_self":   round(p_self,   4),
            "pBayes_friend": round(p_friend, 4),
            "p_bias":        round(p_bias,   4),
            "direction":     direction,
            "evidence":      evidence,
        })

        print(f"    z({cond}): mean={mean_z:.3f}  "
              f"pBayes(self)={p_self:.4f}  "
              f"pBayes(friend)={p_friend:.4f}  "
              f"→ {evidence} {direction}-bias")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        os.path.join(OUTPUT_DIR,
                     f"{exp_label}_{rep_label}_{model_name}_zbias.csv"),
        index=False
    )
    return result_df


# ═══════════════════════════════════════════════════════════════
# STEP 6 — POSTERIOR PREDICTIVE CHECK
# WHY:  Verifies model fit quality by comparing observed vs
#       predicted RT quantiles (.1/.3/.5/.7/.9) — paper p.1471.
#       Good fit = points cluster along the diagonal.
# NEXT: If PPC looks bad, the model may be mis-specified.
# ═══════════════════════════════════════════════════════════════

def run_ppc(model, data, exp_label, rep_label, model_name):
    import hddm

    print(f"\n    Running PPC…")
    try:
        ppc_data  = hddm.utils.post_pred_gen(model, samples=500)
        ppc_stats = hddm.utils.post_pred_stats(data, ppc_data)
        ppc_stats.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"{exp_label}_{rep_label}_{model_name}_PPC.csv")
        )

        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, resp, label in zip(
            axes, [0, 1], ["Friend response", "Self response"]
        ):
            obs  = data[data["response"] == resp]["rt"]
            pred = ppc_data[ppc_data["response"] == resp]["rt"]
            obs_q  = np.quantile(obs,  quantiles)
            pred_q = np.quantile(pred, quantiles)
            lims = [min(obs_q.min(), pred_q.min()) - 0.02,
                    max(obs_q.max(), pred_q.max()) + 0.02]
            ax.plot(obs_q, pred_q, "o-", color="#1D9E75")
            ax.plot(lims, lims, "--", color="#888780", linewidth=0.8)
            ax.set_xlabel("Observed RT quantile (s)")
            ax.set_ylabel("Predicted RT quantile (s)")
            ax.set_title(label, fontsize=11)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        fig.suptitle(
            f"PPC — {exp_label} {rep_label} {model_name}", fontsize=12
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT_DIR,
                         f"{exp_label}_{rep_label}_{model_name}_PPC.png"),
            dpi=150
        )
        plt.close(fig)
        print("    PPC saved.")
    except Exception as e:
        print(f"    ⚠ PPC failed: {e}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — FINAL SUMMARY
# WHY:  Side-by-side comparison of successful vs failed across
#       all experiments — the core deliverable.
# ═══════════════════════════════════════════════════════════════

def build_summary(all_zbias: list):
    summary = pd.concat(all_zbias, ignore_index=True)
    pivot   = summary.pivot_table(
        index=["experiment", "stim_cond"],
        columns="replication",
        values=["mean_z", "p_bias", "evidence"],
        aggfunc="first"
    )

    print("\n\n" + "=" * 60)
    print("  FINAL SUMMARY — Successful vs Failed Replication")
    print("=" * 60)
    print(pivot.to_string())

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "SUMMARY_all_zbias.csv"), index=False
    )
    pivot.to_csv(
        os.path.join(OUTPUT_DIR, "SUMMARY_pivot.csv")
    )
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    return summary


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    all_zbias = []

    for exp_num in [1, 2, 3]:
        exp_label = f"Exp{exp_num}"

        for rep_label in ["successful", "failed"]:

            print(f"\n{'='*60}")
            print(f"  {exp_label} — {rep_label} replication")
            print(f"{'='*60}")

            # STEP 1: Preprocess
            print(f"\n  [Step 1 / 6] Preprocessing…")
            fname = f"exp{exp_num}_{rep_label}.csv"
            raw   = pd.read_csv(os.path.join(INPUT_DIR, fname))
            data  = preprocess(raw, exp_num)

            # STEP 2: Fit models
            print(f"\n  [Step 2 / 6] Fitting HDDM models "
                  f"({len(EXP_MODELS[exp_num])} models)…")
            models_dict = {}
            for model_name, depends_on in EXP_MODELS[exp_num]:
                m = fit_model(
                    data, model_name, depends_on,
                    exp_label, rep_label
                )
                models_dict[model_name] = m

            # STEP 3: DIC comparison
            print(f"\n  [Step 3 / 6] DIC comparison…")
            best_name, _ = compare_dic(
                models_dict, exp_label, rep_label, exp_num
            )
            best_model = models_dict[best_name]

            # STEP 4: Extract parameters
            print(f"\n  [Step 4 / 6] Extracting parameters…")
            extract_parameters(
                best_model, exp_label, rep_label, best_name
            )

            # STEP 5: Z-bias tests
            print(f"\n  [Step 5 / 6] Testing z starting-point bias…")
            zbias = test_z_bias(
                best_model, exp_num,
                exp_label, rep_label, best_name
            )
            all_zbias.append(zbias)

            # STEP 6: PPC
            print(f"\n  [Step 6 / 6] Posterior Predictive Check…")
            run_ppc(best_model, data, exp_label, rep_label, best_name)

    # STEP 7: Final summary
    print(f"\n  [Step 7 / 7] Building final summary table…")
    build_summary(all_zbias)


if __name__ == "__main__":
    main()