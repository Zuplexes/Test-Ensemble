"""เปรียบเทียบ Decision Tree vs Bagging vs Boosting (accuracy + เวลาเทรน)
รัน: python compare.py"""
import time
import numpy as np
from tree import SimpleTree
from bagging import SequentialBagging, ParallelBagging
from boosting import SequentialBoosting

DEPTH, N_TREES, REPEATS = 3, 20, 10     # ทุกโมเดลใช้ depth เท่ากัน, ทำซ้ำ 10 ครั้ง

MODELS = {
    "Decision Tree":        lambda seed: SimpleTree(DEPTH),
    "Bagging (sequential)": lambda seed: SequentialBagging(N_TREES, DEPTH, seed),
    "Bagging (parallel)":   lambda seed: ParallelBagging(N_TREES, DEPTH, seed),
    "Boosting":             lambda seed: SequentialBoosting(N_TREES, DEPTH, seed),
}


def add_noise(y, frac, rng):
    """สลับ label ของชุด train ให้ผิดตามสัดส่วน frac"""
    y, K = y.copy(), y.max() + 1
    i = rng.choice(len(y), int(frac * len(y)), replace=False)
    y[i] = (y[i] + rng.randint(1, K, len(i))) % K
    return y


def run(X, y, noise=0.0):
    """แบ่ง train/test 80/20 แบบสุ่ม ทำซ้ำ REPEATS ครั้ง -> accuracy และเวลาเทรนของแต่ละโมเดล"""
    acc = {name: [] for name in MODELS}
    ms = {name: [] for name in MODELS}
    for r in range(REPEATS):
        rng = np.random.RandomState(r)
        idx = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        train, test = idx[:cut], idx[cut:]
        y_train = add_noise(y[train], noise, rng)       # noise ใส่เฉพาะ train

        for name, make in MODELS.items():
            model = make(r)
            t = time.time()
            model.fit(X[train], y_train)
            ms[name].append((time.time() - t) * 1000)
            acc[name].append(np.mean(model.predict(X[test]) == y[test]) * 100)  # test ใช้ label จริง
    return acc, ms


def show(title, acc, ms):
    print(f"\n{title}")
    print(f"{'Model':22s} {'Accuracy (%)':>16s} {'Train time (ms)':>17s}")
    for name in MODELS:
        print(f"{name:22s} {np.mean(acc[name]):9.2f} ± {np.std(acc[name]):4.2f} {np.mean(ms[name]):15.1f}")


def plot(clean, noisy, ms):
    import matplotlib.pyplot as plt
    names, x = list(MODELS), np.arange(len(MODELS))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for dx, (label, acc) in zip((-0.2, 0.2), (("clean data", clean), ("20% label noise", noisy))):
        ax[0].bar(x + dx, [np.mean(acc[n]) for n in names], 0.4,
                  yerr=[np.std(acc[n]) for n in names], capsize=3, label=label)
    ax[0].set(title="Accuracy (test 20%)", ylabel="accuracy (%)", ylim=(60, 100))
    ax[0].legend()

    ax[1].bar(x, [np.mean(ms[n]) for n in names], color="tab:green")
    ax[1].set(title="Training time (clean data)", ylabel="milliseconds")

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=15); a.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved results.png")
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_iris      # import ในนี้ เพื่อไม่ให้ worker ต้องโหลดซ้ำ
    X, y = load_iris(return_X_y=True)
    print(f"Iris: train 120 / test 30 | depth={DEPTH}, trees={N_TREES}, ทำซ้ำ {REPEATS} ครั้ง")

    clean, ms = run(X, y, noise=0.0)
    noisy, _ = run(X, y, noise=0.2)
    show("ข้อมูลปกติ", clean, ms)
    show("Train มี label ผิด 20%", noisy, _)
    plot(clean, noisy, ms)