"""
test_sqa.py
Verifica il comportamento della SQA su ECG reali, per ogni classe del dataset.

Pipeline applicata: notch -> bandpass -> downsample (identica a build_dataset.py)
ma senza z-score, perche' la SQA lavora su ampiezze assolute.

SEMANTICA DEL TEST
==================
  REJECT_CLASSES (RL-RA, RL-LA): la SQA DEVE scartarle al 100%.
    - [OK]   se valid=False  (correctly rejected)
    - [FAIL] se valid=True   (missed rejection — critico)
    → Il test FALLISCE se anche un solo ECG di queste classi passa la SQA.

  ACCEPT_CLASSES (tutte le altre): la SQA NON DEVE over-rejectare.
    - Un singolo rifiuto e' legittimo se il segnale e' davvero rumoroso.
    - [OK]   se valid=True
    - [INFO] se valid=False (segnale singolo rifiutato, motivo stampato)
    - [WARN] se reject_rate > REJECT_WARN_THR (default 50%)
    - [FAIL] se reject_rate > REJECT_FAIL_THR (default 80%)

USO
===
  CWD atteso: elettrodo_inverso/src/
  python -m tests.test_sqa
  python -m tests.test_sqa --verbose    # stampa feature raw per ogni finestra problematica
  python -m tests.test_sqa --n 10       # usa 10 ECG per classe (default 5)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing, check_ecg_quality, check_lead_quality_global
from utils.config import ALL_LEADS, LIMB_LEADS, QUALITY_CFG, LABEL_MAP_CLEAN, FLATLINE_CLASSES

# --- CONFIG ---
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'thesis-sample.csv')
LIMB_INDICES   = list(range(6))   # I, II, III, aVr, aVl, aVf
REJECT_WARN_THR = 0.50   # >50% rifiuti su classi 0-5 → WARN
REJECT_FAIL_THR = 0.80   # >80% rifiuti su classi 0-5 → FAIL


def run_sqa_on_ecg(ecg_id):
    """
    Carica l'EDF, applica preprocessing (senza z-score) e lancia check_ecg_quality
    solo sulle derivazioni periferiche (lead 0..5, che sono quelle impattate dalle
    inversioni limb e in particolare da RL-RA / RL-LA).
    Ritorna (result_dict, sigs_array) oppure (None, None) se non leggibile.
    """
    ecg_data = get_ecg(ecg_id)
    if not ecg_data or not ecg_data["signals"]:
        return None, None

    sigs = all_leads_preprocessing(ecg_data["signals"])
    missing = [l for l in ALL_LEADS if l not in sigs]
    if missing:
        return None, None

    sigs_array = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
    if sigs_array.shape[0] != 12:
        return None, None

    result = check_ecg_quality(sigs_array, cfg=QUALITY_CFG, lead_indices=LIMB_INDICES)
    return result, sigs_array


def print_lead_features(sigs_array, lead_indices=LIMB_INDICES):
    """Stampa le feature globali per ogni derivazione (utile in modalita' verbose)."""
    print("    [features per derivazione]")
    for i in lead_indices:
        r = check_lead_quality_global(sigs_array[i], cfg=QUALITY_CFG, lead_idx=i)
        lead_name = ALL_LEADS[i]
        valid_str = "OK " if r["valid"] else "BAD"
        print(
            f"      {lead_name:5s} [{valid_str}]  "
            f"std={r['std']:7.2f}  ptp={r['ptp']:7.2f}  "
            f"median_abs={r['median_abs']:7.2f}  mad_diff={r['mad_diff']:6.3f}  "
            f"reason={r['reason']}"
        )


def test_sqa_per_class(n_per_class=5, verbose=False):
    if not os.path.exists(CSV_PATH):
        print(f"[ERRORE] CSV non trovato: {CSV_PATH}")
        print(f"  Percorso atteso: {os.path.abspath(CSV_PATH)}")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df["Inversione"] != "?"].copy()
    df["label"] = df["Inversione"].map(LABEL_MAP_CLEAN)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str)

    classes = sorted(df["label"].unique())
    print(f"Classi nel dataset: {classes}")
    print(f"REJECT_CLASSES (SQA deve scartare al 100%): {FLATLINE_CLASSES}")
    print(f"Testing {n_per_class} ECG per classe...\n")

    results = {}  # label -> dict con contatori

    for label in classes:
        subset  = df[df["label"] == label]
        sample  = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        is_reject_class = label in FLATLINE_CLASSES

        ok = fail = info_rejected = missed_rejection = 0
        print(f"{'='*65}")
        print(f"  Classe: {label}  (n={len(sample)}, "
              f"aspettativa: {'RIFIUTO 100%' if is_reject_class else 'ACCETTAZIONE (tolleranza rumore)'})")
        print(f"{'='*65}")

        for _, row in sample.iterrows():
            ecg_id = row["Num"]
            result, sigs_array = run_sqa_on_ecg(ecg_id)

            if result is None:
                print(f"  ECG {ecg_id}: [SKIP] non leggibile")
                continue

            valid       = result["global_valid"]
            n_valid     = result["valid_leads"]
            total_leads = result["total_leads"]
            invalid_leads = [ALL_LEADS[i] for i in result["invalid_lead_indices"]
                             if i < len(ALL_LEADS)]

            if is_reject_class:
                # Deve essere RIFIUTATO
                if not valid:
                    missed_rejection_delta = 0
                    ok += 1
                    icon = "[OK  ]"
                else:
                    missed_rejection += 1
                    fail += 1
                    icon = "[FAIL]"
            else:
                # Dovrebbe essere accettato (ma un rifiuto motivato e' tollerato)
                if valid:
                    ok += 1
                    icon = "[OK  ]"
                else:
                    info_rejected += 1
                    icon = "[INFO]"

            reason_str = ""
            if not valid:
                # Raccoglie i motivi aggregati dalle singole lead
                reasons = set()
                for r in result["lead_results"]:
                    if not r["valid"]:
                        reasons.update(r.get("global_result", {}).get("reason", "?").split(", "))
                reason_str = f"  motivo: {', '.join(sorted(reasons))}"

            lead_info = f"  bad_leads: [{', '.join(invalid_leads)}]" if invalid_leads else ""
            print(f"  ECG {ecg_id}: {icon}  valid={str(valid):5s}  leads={n_valid}/{total_leads}"
                  f"{lead_info}{reason_str}")

            if verbose and sigs_array is not None:
                print_lead_features(sigs_array)

        print()
        results[label] = {
            "ok": ok, "fail": fail,
            "info_rejected": info_rejected,
            "missed_rejection": missed_rejection,
            "is_reject_class": is_reject_class,
            "total": ok + fail + info_rejected,
        }

    # --- RIEPILOGO ---
    print("=" * 65)
    print("RIEPILOGO SQA PER CLASSE")
    print("=" * 65)
    print(f"{'Classe':20s}  {'Risultato':38s}  Status")
    print("-" * 75)

    global_pass = True
    for label, r in results.items():
        tot = r["total"]
        if tot == 0:
            print(f"  {label:20s}  {'nessun ECG caricato':38s}  [SKIP]")
            continue

        if r["is_reject_class"]:
            reject_rate = (r["ok"]) / tot   # ok = correttamente rifiutati
            if r["missed_rejection"] == 0:
                icon = "[PASS]"
                desc = f"rifiutati {r['ok']}/{tot} (100%) - CORRETTO"
            else:
                icon = "[FAIL]"
                desc = f"rifiutati {r['ok']}/{tot}, MANCATI {r['missed_rejection']}"
                global_pass = False
        else:
            reject_rate = r["info_rejected"] / tot
            accepted    = r["ok"]
            if reject_rate == 0:
                icon = "[PASS]"
                desc = f"accettati {accepted}/{tot} (0% falsi rifiuti)"
            elif reject_rate <= REJECT_WARN_THR:
                icon = "[WARN]"
                desc = (f"accettati {accepted}/{tot}, "
                        f"rifiutati {r['info_rejected']} ({reject_rate:.0%})")
            elif reject_rate <= REJECT_FAIL_THR:
                icon = "[WARN]"
                desc = (f"accettati {accepted}/{tot}, "
                        f"rifiutati {r['info_rejected']} ({reject_rate:.0%}) - HIGH-REJECT")
            else:
                icon = "[FAIL]"
                desc = (f"accettati {accepted}/{tot}, "
                        f"rifiutati {r['info_rejected']} ({reject_rate:.0%}) - OVER-REJECT")
                global_pass = False

        print(f"  {label:20s}  {desc:38s}  {icon}")

    print("-" * 75)
    status = "PASSA [v]" if global_pass else "HA PROBLEMI [x]"
    print(f"\nSQA complessiva: {status}")
    print()
    if not global_pass:
        print("AZIONI CORRETTIVE:")
        print("  Se classi 6/7 sfuggono: abbassare QUALITY_CFG['near_zero_median_thr']")
        print("                          o abbassare min_mad_diff_limb")
        print("  Se classi 0-5 over-reject: alzare near_zero_median_thr o std_low_limb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SQA per classe ECG")
    parser.add_argument("--verbose", action="store_true",
                        help="Stampa feature raw per ogni derivazione")
    parser.add_argument("--n", type=int, default=5, metavar="N",
                        help="Numero di ECG da testare per classe (default: 5)")
    args = parser.parse_args()
    test_sqa_per_class(n_per_class=args.n, verbose=args.verbose)
