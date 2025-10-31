"""
Fisher's Exact Test + Fisher's Exact Test for Randomization — Significance of CoT Explanatory Power
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# ---------------------------- Configuration Zone ----------------------------
np.random.seed(42)
n_sample = 200          # Total number of samples per task
n_perm   = 10_000       # Number of randomizations
# --------------------------------------------------------------

tasks_acc = {
    "LastLetter":   (0.85, 0.10),
    "CircuitValue": (0.95, 0.15),
    "5×5 Multiplication":(0.964, 0.019),
    "MultiArith":   (0.80, 0.20),
    "CoinFlip":     (0.95, 0.50),
    "GSM8K":        (0.659, 0.354),
    "AQuA":         (0.58, 0.40),
    "GSM8K-Aug":    (0.219, 0.17),
    "Parity":       (0.95, 0.50),
    "Llama-3-8B Arithmetic":   (0.80, 0.42),
}

results = []

for task, (acc_cot, acc_no) in tasks_acc.items():
    # ---------- 1. Construct sample-level labels ----------
    n_cot = n_no = n_sample // 2
    y_cot = np.concatenate([np.ones(int(acc_cot * n_cot)),
                            np.zeros(n_cot - int(acc_cot * n_cot))])
    y_no  = np.concatenate([np.ones(int(acc_no * n_no)),
                            np.zeros(n_no - int(acc_no * n_no))])
    y = np.concatenate([y_cot, y_no])
    trt = np.array([1]*n_cot + [0]*n_no)

    # ---------- 2. Fisher's Exact Test ----------
    a, b = y_cot.sum(), (1-y_cot).sum()
    c, d = y_no.sum(),  (1-y_no).sum()
    table = [[a, b], [c, d]]
    _, p_exact = fisher_exact(table, alternative='greater')

    # ---------- 3. Fisher's Exact Test ----------
    w_obs = (acc_cot - acc_no) / acc_no
    w_perm = []
    for _ in range(n_perm):
        trt_shuf = np.random.permutation(trt)
        acc1 = y[trt_shuf==1].mean()
        acc0 = y[trt_shuf==0].mean()
        w_perm.append((acc1 - acc0) / acc0)
    p_rand = (np.array(w_perm) >= w_obs).mean()

    # ---------- 4. Asterisk Marking ----------
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    results.append({"Task": task, "p_exact": p_exact, "p_rand": p_rand,
                    "sig_exact": stars(p_exact), "sig_rand": stars(p_rand)})

df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.4f"))
