"""Fit GMM trên toàn bộ chiều embedding raw, xuất 6 biểu đồ PDF riêng."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.mixture import GaussianMixture


def load_samples(root, subsets, sides, dirs_per_subset, max_total, seed):
    rng = np.random.default_rng(seed)
    chunks = []

    for subset in subsets:
        path = Path(root) / subset
        names = sorted(p for p in path.iterdir() if p.is_dir())
        if dirs_per_subset > 0 and len(names) > dirs_per_subset:
            indices = rng.choice(len(names), dirs_per_subset, replace=False)
            names = [names[i] for i in indices]

        rows = []
        for directory in names:
            for side in sides:
                rep = torch.load(directory / f"{side}.pt", map_location="cpu", weights_only=False)["rep"]
                rows.append(rep.float().reshape(-1, rep.shape[-1]).numpy())

        if not rows:
            raise ValueError(f"Không có embedding trong {path}")

        sub = np.concatenate(rows, axis=0)
        print(f"{subset}: {len(names)} dirs -> {len(sub)} embeddings")
        chunks.append(sub)

    X = np.concatenate(chunks, axis=0)
    if max_total > 0 and len(X) > max_total:
        X = X[rng.choice(len(X), max_total, replace=False)]

    X = X.astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Embedding chứa NaN hoặc Inf")
    return X


def marginal_variances(gmm, dim):
    """Variance của từng component tại một chiều gốc."""
    cov = gmm.covariances_
    if gmm.covariance_type == "full":
        return cov[:, dim, dim]
    if gmm.covariance_type == "tied":
        return np.full(gmm.n_components, cov[dim, dim])
    if gmm.covariance_type == "diag":
        return cov[:, dim]
    return cov  # spherical


def projected_variances(gmm, direction):
    """Variance của từng component sau khi chiếu lên direction."""
    cov = gmm.covariances_
    if gmm.covariance_type == "full":
        return np.einsum("d,kde,e->k", direction, cov, direction)
    if gmm.covariance_type == "tied":
        return np.full(gmm.n_components, direction @ cov @ direction)
    if gmm.covariance_type == "diag":
        return cov @ direction.square() if hasattr(direction, "square") else cov @ (direction ** 2)
    return cov * np.dot(direction, direction)


def mixture_pdf(grid, weights, means, variances):
    stds = np.sqrt(np.maximum(variances, np.finfo(np.float64).tiny))
    return (stats.norm.pdf(grid[:, None], loc=means[None, :], scale=stds[None, :]) * weights[None, :]).sum(axis=1)


def mahalanobis_squared(X, gmm):
    """Mahalanobis² dưới GMM một component, không tạo covariance full cho diag."""
    diff = X - gmm.means_[0]
    precision_chol = gmm.precisions_cholesky_

    if gmm.covariance_type == "full":
        whitened = diff @ precision_chol[0]
    elif gmm.covariance_type == "tied":
        whitened = diff @ precision_chol
    elif gmm.covariance_type == "diag":
        whitened = diff * precision_chol[0]
    else:
        whitened = diff * precision_chol[0]

    return np.einsum("nd,nd->n", whitened, whitened)


def save_figure(fig, output_dir, filename):
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def fit_gmm(X, k, args, n_init):
    gmm = GaussianMixture(
        n_components=k, covariance_type=args.cov_type,
        reg_covar=args.reg_covar, n_init=n_init,
        max_iter=args.max_iter, random_state=args.seed,
    ).fit(X)
    print(f"K={k}: converged={gmm.converged_}, n_iter={gmm.n_iter_}")
    return gmm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caching_dir", required=True)
    ap.add_argument("--subset_name", nargs="+", required=True)
    ap.add_argument("--side", choices=["qry", "pos", "both"], default="qry")
    ap.add_argument("--dirs_per_subset", type=int, default=2000, help="<=0: tất cả thư mục")
    ap.add_argument("--max_total", type=int, default=20000, help="<=0: tất cả embedding đã đọc")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--cov_type", choices=["full", "diag", "tied", "spherical"], default="diag")
    ap.add_argument("--reg_covar", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--out_dir", default="gmm_raw_plots")
    args = ap.parse_args()

    if args.K < 1 or args.k_max < 1 or args.reg_covar <= 0:
        raise ValueError("K, k_max >= 1 và reg_covar > 0")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})

    sides = ["qry", "pos"] if args.side == "both" else [args.side]
    X = load_samples(args.caching_dir, args.subset_name, sides, args.dirs_per_subset, args.max_total, args.seed)

    if args.dedup:
        n0 = len(X)
        _, indices = np.unique(np.round(X, 5), axis=0, return_index=True)
        X = X[np.sort(indices)]
        print(f"dedup: {n0} -> {len(X)}")

    N, D = X.shape
    if D < 2 or N < max(2, args.K, args.k_max):
        raise ValueError("Cần ít nhất 2 chiều và đủ sample để fit các GMM")

    # Chọn chiều gốc; không PCA, không normalize.
    variances = X.var(axis=0)
    dims = np.argsort(-variances, kind="stable")[:2]
    print(f"X: {X.shape}")
    print(f"Top variance dimensions (0-based): {dims.tolist()}")
    print(f"Variances: {variances[dims].tolist()}")

    # Fit trên toàn bộ D chiều.
    g1 = fit_gmm(X, 1, args, n_init=1)
    gk = g1 if args.K == 1 else fit_gmm(X, args.K, args, n_init=5)
    print(f"K=1: BIC={g1.bic(X):.1f}, avg loglik={g1.score(X):.3f}")
    print(f"K={args.K}: BIC={gk.bic(X):.1f}, avg loglik={gk.score(X):.3f}")

    # 1–2: Marginal trên hai chiều gốc có variance lớn nhất.
    for rank, dim in enumerate(dims, start=1):
        values = X[:, dim]
        grid = np.linspace(values.min(), values.max(), 500)
        normal = mixture_pdf(grid, g1.weights_, g1.means_[:, dim], marginal_variances(g1, dim))
        mixture = mixture_pdf(grid, gk.weights_, gk.means_[:, dim], marginal_variances(gk, dim))

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(values, bins=args.bins, density=True, alpha=0.4, color="steelblue", label="Embeddings")
        ax.plot(grid, normal, color="crimson", label="Single Gaussian")
        ax.plot(grid, mixture, color="forestgreen", label=f"GMM (K={args.K})")
        ax.set_xlabel(f"Embedding dimension {dim}")
        ax.set_ylabel("Density")
        ax.set_title(f"Variance rank {rank}")
        ax.legend()
        save_figure(fig, output_dir, f"0{rank}_marginal_dim_{dim}.pdf")

    # 3: Chiếu lên hướng nối hai tâm component xa nhất.
    distance = np.linalg.norm(gk.means_[:, None, :] - gk.means_[None, :, :], axis=-1)
    direction = np.zeros(D)
    direction[dims[0]] = 1.0

    if args.K > 1:
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        delta = gk.means_[i] - gk.means_[j]
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-12:
            direction = delta / delta_norm

    values = X @ direction
    grid = np.linspace(values.min(), values.max(), 500)
    normal = mixture_pdf(grid, g1.weights_, g1.means_ @ direction, projected_variances(g1, direction))
    mixture = mixture_pdf(grid, gk.weights_, gk.means_ @ direction, projected_variances(gk, direction))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(values, bins=args.bins, density=True, alpha=0.4, color="steelblue", label="Embeddings")
    ax.plot(grid, normal, color="crimson", label="Single Gaussian")
    ax.plot(grid, mixture, color="forestgreen", label=f"GMM (K={args.K})")
    ax.set_xlabel("Projected embedding value")
    ax.set_ylabel("Density")
    ax.set_title("Farthest-centroid direction")
    ax.legend()
    save_figure(fig, output_dir, "03_centroid_direction.pdf")

    # 4: Mahalanobis² dưới Gaussian một component.
    m2 = mahalanobis_squared(X, g1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    stats.probplot(m2, dist=stats.chi2(D), plot=ax)
    ax.set_title(f"Mahalanobis squared vs. chi-square ({D} dimensions)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed squared distances")
    save_figure(fig, output_dir, "04_mahalanobis_qq.pdf")

    # 5: BIC theo K, tất cả đều fit trên toàn bộ D chiều.
    ks = list(range(1, args.k_max + 1))
    bics = []
    for k in ks:
        if k == 1:
            fitted = g1
        elif k == args.K:
            fitted = gk
        else:
            fitted = fit_gmm(X, k, args, n_init=5)
        bics.append(fitted.bic(X))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, bics, "o-", color="steelblue")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of GMM components")
    ax.set_ylabel("BIC")
    ax.set_title("BIC on full-dimensional embeddings")
    save_figure(fig, output_dir, "05_bic_vs_k.pdf")

    # 6: Scatter của hai chiều variance lớn nhất.
    # Màu cụm được suy ra từ GMM fit trên toàn bộ chiều.
    labels = gk.predict(X)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(X[:, dims[0]], X[:, dims[1]], c=labels, s=3, alpha=0.4, cmap="tab10", rasterized=True)
    ax.set_xlabel(f"Embedding dimension {dims[0]}")
    ax.set_ylabel(f"Embedding dimension {dims[1]}")
    ax.set_title(f"Top-variance dimensions, GMM K={args.K}")
    fig.colorbar(scatter, ax=ax, label="Component", ticks=np.unique(labels))
    save_figure(fig, output_dir, "06_top_variance_scatter.pdf")


if __name__ == "__main__":
    main()