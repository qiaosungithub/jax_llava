import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOKENS = np.array([2**k for k in range(2,9)], dtype=int)
K_VALUES = np.arange(9)


def sample_discrete_logit_normal(mu: float, sigma: float, n_samples: int, rng: np.random.Generator):
    z = rng.normal(loc=mu, scale=sigma, size=n_samples)
    u = 1.0 / (1.0 + np.exp(-z))
    k = np.clip(np.rint(u * len(TOKENS)), 0, len(TOKENS) - 1).astype(int)
    token_counts = TOKENS[k]

    probs = np.bincount(k, minlength=len(TOKENS)).astype(float)
    probs /= probs.sum()
    return token_counts, probs


def plot_examples(output_path: Path):
    rng = np.random.default_rng(20260423)
    n_samples = 300000

    # Larger mu -> favors larger k (more tokens).
    # Larger sigma -> broader spread.
    configs = [
        (-0.0, 1.2),
        # (-0.0, 0.8),
        (-0.0, 0.6),
        # (-0.2, 0.8),
        (1.0, 0.6),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.flatten()

    for i, (mu, sigma) in enumerate(configs):
        label = f"mu={mu:.1f}, sigma={sigma:.1f}"
        token_counts, probs = sample_discrete_logit_normal(mu, sigma, n_samples, rng)
        mean_tokens = token_counts.mean()
        std_tokens = token_counts.std(ddof=0)

        ax = axes[i]
        ax.bar(np.arange(len(TOKENS)), probs, color="#4C72B0", alpha=0.9)
        ax.set_xticks(np.arange(len(TOKENS)))
        ax.set_xticklabels([str(t) for t in TOKENS], rotation=0)
        ax.set_ylim(0, max(0.38, probs.max() * 1.15))
        ax.set_title(
            f"{label}\nE[tokens]={mean_tokens:.1f}, Std={std_tokens:.1f}",
            fontsize=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if i % 3 == 0:
            ax.set_ylabel("Probability")
        if i >= 3:
            ax.set_xlabel("Token count (2^k)")

    # Use the last panel for a concise reminder of mapping.
    legend_ax = axes[-1]
    legend_ax.axis("off")
    lines = [
        "Discrete support:",
        "tokens in {1, 2, 4, ..., 256}",
        "Sampling:",
        "z ~ N(mu, sigma^2)",
        "u = sigmoid(z)",
        "k = round(8u), tokens = 2^k",
    ]
    legend_ax.text(0.03, 0.95, "\n".join(lines), va="top", fontsize=11)

    fig.suptitle("Discrete Logit-Normal Distributions over Token Budget", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    output_path = Path(__file__).resolve().parent / "discrete_logit_normal_examples.png"
    plot_examples(output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
