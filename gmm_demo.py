"""Demo nhanh: lấy mẫu ngẫu nhiên embedding từ cache -> PCA -> GMM (K=1 vs K) -> plot."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def load_samples(root, subsets, sides, dirs_per_subset, max_total, seed):
    """Chọn ngẫu nhiên dirs_per_subset thư mục mẫu mỗi subset, đọc rep, rồi cắt ngẫu nhiên còn max_total dòng."""
    rng = np.random.default_rng(seed)
    chunks = []
    for subset in subsets:
        path = os.path.join(root, subset)
        names = sorted(n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)))
        if dirs_per_subset > 0 and len(names) > dirs_per_subset:
            names = list(rng.choice(names, dirs_per_subset, replace=False))
        rows = []
        for n in names:
            for side in sides:
                rep = torch.load(os.path.join(path, n, f"{side}.pt"),
                                 map_location="cpu", weights_only=False)["rep"]
                rows.append(rep.float().reshape(-1, rep.shape[-1]).numpy())
        sub = np.concatenate(rows)
        print(f"{subset}: {len(names)} dirs -> {len(sub)} embeddings")
        chunks.append(sub)
    X = np.concatenate(chunks).astype(np.float64)
    if len(X) > max_total:
        X = X[rng.choice(len(X), max_total, replace=False)]
    return X


def full_cov(g):
    """Đổi covariances_ của mọi cov_type về shape (K, D, D)."""
    K, D = g.n_components, g.means_.shape[1]
    c = g.covariances_
    if g.covariance_type == "full":
        return c
    if g.covariance_type == "tied":
        return np.repeat(c[None], K, axis=0)
    if g.covariance_type == "diag":
        return np.stack([np.diag(v) for v in c])
    return np.stack([np.eye(D) * v for v in c])  # spherical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caching_dir", required=True)
    ap.add_argument("--subset_name", nargs="+", required=True)
    ap.add_argument("--side", choices=["qry", "pos", "both"], default="qry")
    ap.add_argument("--dirs_per_subset", type=int, default=2000, help="số thư mục mẫu ngẫu nhiên mỗi subset (<=0: tất cả)")
    ap.add_argument("--max_total", type=int, default=20000, help="tối đa số embedding dùng để fit")
    ap.add_argument("--dedup", action="store_true", help="loại embedding trùng lặp trước khi PCA")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--cov_type", choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="gmm_demo.png")
    args = ap.parse_args()

    sides = ["qry", "pos"] if args.side == "both" else [args.side]
    X = load_samples(args.caching_dir, args.subset_name, sides,
                     args.dirs_per_subset, args.max_total, args.seed)
    if args.dedup:
        n0 = len(X)
        X = np.unique(np.round(X, 5), axis=0)
        print(f"dedup: {n0} -> {len(X)}")
    print("X:", X.shape)

    # ---- PCA + GMM ----
    Z = PCA(args.pca_dim, random_state=args.seed).fit_transform(X)
    D, K = Z.shape[1], args.K
    g1 = GaussianMixture(1, covariance_type=args.cov_type, reg_covar=1e-4, 
                         random_state=args.seed, verbose=2, verbose_interval=1,).fit(Z)
    gk = GaussianMixture(K, covariance_type=args.cov_type, reg_covar=1e-4, n_init=5,
                         random_state=args.seed, verbose=2, verbose_interval=1,).fit(Z)
    print(f"K=1 : BIC={g1.bic(Z):.1f}  avg loglik={g1.score(Z):.3f}")
    print(f"K={K} : BIC={gk.bic(Z):.1f}  avg loglik={gk.score(Z):.3f}")

    covk, cov1 = full_cov(gk), full_cov(g1)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # 1) PC1, PC2: histogram + 1 Gaussian vs GMM marginal
    for d in range(2):
        t = np.linspace(Z[:, d].min(), Z[:, d].max(), 400)
        ax[0, d].hist(Z[:, d], bins=80, density=True, alpha=.4)
        ax[0, d].plot(t, stats.norm.pdf(t, Z[:, d].mean(), Z[:, d].std()), "r", label="1 Gaussian")
        mix = sum(w * stats.norm.pdf(t, m[d], np.sqrt(c[d, d]))
                  for w, m, c in zip(gk.weights_, gk.means_, covk))
        ax[0, d].plot(t, mix, "g", label=f"GMM K={K}")
        ax[0, d].set_title(f"PC{d+1}")
        ax[0, d].legend()

    # 2) chiếu lên hướng nối hai tâm xa nhất
    dist = np.linalg.norm(gk.means_[:, None] - gk.means_[None], axis=-1)
    if K > 1:
        i, j = np.unravel_index(np.argmax(dist), (K, K))
        u = gk.means_[i] - gk.means_[j]
        u /= np.linalg.norm(u)
    else:
        u = np.eye(D)[0]
    p = Z @ u
    t = np.linspace(p.min(), p.max(), 400)
    ax[0, 2].hist(p, bins=80, density=True, alpha=.4)
    ax[0, 2].plot(t, stats.norm.pdf(t, p.mean(), p.std()), "r", label="1 Gaussian")
    mix = sum(w * stats.norm.pdf(t, m @ u, np.sqrt(u @ c @ u))
              for w, m, c in zip(gk.weights_, gk.means_, covk))
    ax[0, 2].plot(t, mix, "g", label=f"GMM K={K}")
    ax[0, 2].set_title("hướng nối 2 tâm cụm xa nhất")
    ax[0, 2].legend()

    # 3) Mahalanobis^2 dưới 1 Gaussian vs chi2_D
    diff = Z - g1.means_[0]
    m2 = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov1[0]), diff)
    stats.probplot(m2, dist=stats.chi2(D), plot=ax[1, 0])
    ax[1, 0].set_title(f"Mahalanobis² vs χ²_{D} (fit 1 Gaussian)")

    # 4) BIC theo K
    ks = list(range(1, args.k_max + 1))
    bics = [GaussianMixture(k, covariance_type=args.cov_type, reg_covar=1e-4, n_init=2,
                            random_state=args.seed).fit(Z).bic(Z) for k in ks]
    ax[1, 1].plot(ks, bics, "o-")
    ax[1, 1].set_xlabel("K")
    ax[1, 1].set_title("BIC vs K")

    # 5) PCA 2D tô màu theo cụm GMM
    ax[1, 2].scatter(Z[:, 0], Z[:, 1], c=gk.predict(Z), s=2, alpha=.4, cmap="tab10")
    ax[1, 2].set_title(f"PCA 2D, màu theo cụm GMM (K={K})")

    fig.suptitle(f"N={len(Z)} | PCA {D}d | cov={args.cov_type} | subsets={args.subset_name} | side={args.side}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    
    print("Saved", args.out)


if __name__ == "__main__":
    main()