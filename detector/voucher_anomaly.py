
"""
CAG 101: Graph-Spectral Voucher Anomaly Detection

Pipeline:
1. Prepare date and amount features.
2. Construct a Gaussian all-pairs similarity graph.
3. Calculate diffusion and symmetric spectral operators.
4. Apply a Morlet-type graph spectral filter at multiple scales.
5. Calculate E (wavelet energy), Q (diffusion-space isolation),
   and R (feature reconstruction residual).
6. Combine the three components into an anomaly score.

Research note:
The definitions of Q, R, feature construction and scale weighting
below are explicit prototype choices. Verify them against the
equations in the original project document before claiming that
this implementation exactly reproduces that formulation.
"""

import numpy as np
import pandas as pd

from scipy.linalg import eigh


def _minmax(values):
    """Normalize finite numeric values to [0, 1]."""
    values = np.asarray(values, dtype=float)
    output = np.zeros_like(values)

    finite = np.isfinite(values)

    if not finite.any():
        return output

    low = values[finite].min()
    high = values[finite].max()

    if high > low:
        output[finite] = (
            (values[finite] - low) / (high - low)
        )

    return output


def _prepare_features(df, date_column, amount_column):
    """Convert date and amount into two standardized features."""

    result = df.copy()

    dates = pd.to_datetime(
        result[date_column],
        errors="coerce",
        dayfirst=True,
    )

    amount_text = (
        result[amount_column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )

    amounts = pd.to_numeric(
        amount_text,
        errors="coerce",
    )

    if amounts.notna().sum() < 2:
        raise ValueError(
            "At least two valid numeric transaction amounts are required."
        )

    # Represent dates as days relative to the earliest valid date.
    valid_dates = dates.dropna()

    if valid_dates.empty:
        raise ValueError(
            "At least one valid transaction date is required."
        )

    date_days = (
        (dates - valid_dates.min()).dt.total_seconds()
        / 86400.0
    )

    features = pd.DataFrame({
        "amount": amounts,
        "date_days": date_days,
    })

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation.
    for column in features.columns:
        median = features[column].median()

        if pd.isna(median):
            raise ValueError(
                f"Feature '{column}' contains no valid values."
            )

        features[column] = features[column].fillna(median)

    X = features.to_numpy(dtype=float)

    # Standardize each feature.
    means = X.mean(axis=0)
    standard_deviations = X.std(axis=0)

    standard_deviations[standard_deviations == 0] = 1.0

    X_scaled = (X - means) / standard_deviations

    result["parsed_date"] = dates
    result["parsed_amount"] = amounts

    return result, X_scaled


def _build_gaussian_graph(X, epsilon=1.0):
    """
    Build an all-pairs Gaussian affinity graph.

    W_ij = exp(-d_ij^2 / (4 epsilon))

    The diagonal is set to zero so a transaction does not
    count itself as a graph neighbour.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    differences = X[:, None, :] - X[None, :, :]

    distances_squared = np.sum(
        differences ** 2,
        axis=2,
    )

    W = np.exp(
        -distances_squared / (4.0 * epsilon)
    )

    np.fill_diagonal(W, 0.0)

    # Ensure numerical symmetry.
    W = (W + W.T) / 2.0

    return W, distances_squared


def _spectral_decomposition(W):
    """
    Construct:
        P = D^(-1) W
        S = D^(-1/2) W D^(-1/2)

    Return the eigenvalues and eigenvectors of symmetric S.
    """

    degree = W.sum(axis=1)

    if np.any(degree <= 0):
        raise ValueError(
            "The graph contains a node with zero weighted degree."
        )

    # Diffusion operator.
    P = W / degree[:, None]

    # Symmetric operator similar to P.
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)

    S = (
        inverse_sqrt_degree[:, None]
        * W
        * inverse_sqrt_degree[None, :]
    )

    # Normalized symmetric graph Laplacian.
    L = np.eye(len(W)) - S

    eigenvalues, eigenvectors = eigh(S)

    # Frequencies associated with the symmetric operator.
    frequencies = 1.0 - eigenvalues

    # Remove tiny negative frequencies due to numerical errors.
    frequencies = np.clip(frequencies, 0.0, None)

    return (
        P,
        S,
        L,
        degree,
        eigenvalues,
        eigenvectors,
        frequencies,
    )


def _calculate_wavelet_energy(
    X,
    eigenvectors,
    frequencies,
    scales=6,
    omega0=6.0,
    beta=1.0,
    base_scale=10.5,
):
    """
    Calculate multiscale Morlet-type graph wavelet energy.

    g_s(mu) =
        exp(-(s*mu - omega0)^2 / (2*beta^2))
        - exp(-omega0^2 / (2*beta^2))
          * exp(-(s*mu)^2 / (2*beta^2))

    The scale range is logarithmically spaced around base_scale.
    """

    if scales < 1:
        raise ValueError("scales must be at least 1.")

    if beta <= 0 or base_scale <= 0:
        raise ValueError("beta and base_scale must be positive.")

    # Transform each feature signal into the eigenvector basis.
    spectral_coefficients = eigenvectors.T @ X

    scale_values = np.geomspace(
        base_scale / 2.0,
        base_scale * 2.0,
        num=int(scales),
    )

    energy_by_scale = []

    for scale in scale_values:
        scaled_frequency = scale * frequencies

        first_term = np.exp(
            -0.5
            * ((scaled_frequency - omega0) / beta) ** 2
        )

        correction = (
            np.exp(-0.5 * (omega0 / beta) ** 2)
            * np.exp(
                -0.5 * (scaled_frequency / beta) ** 2
            )
        )

        response = first_term - correction

        # Apply spectral response and return to transaction space.
        wavelet_coefficients = eigenvectors @ (
            response[:, None] * spectral_coefficients
        )

        # Sum energy across the available features.
        scale_energy = np.sum(
            np.abs(wavelet_coefficients) ** 2,
            axis=1,
        )

        energy_by_scale.append(scale_energy)

    energy_by_scale = np.asarray(energy_by_scale)

    # Prototype aggregation: equal weight for each scale.
    wavelet_energy = energy_by_scale.mean(axis=0)

    return wavelet_energy, scale_values, energy_by_scale


def _calculate_diffusion_isolation(
    X,
    degree,
    eigenvalues,
    eigenvectors,
    diffusion_time=1,
):
    """
    Prototype definition of Q:
    isolation in a diffusion-coordinate embedding.

    This is an explicit implementation choice; replace it if
    the original document defines Q differently.
    """

    if diffusion_time < 0:
        raise ValueError("diffusion_time cannot be negative.")

    # Skip the largest eigenvalue, which usually represents
    # the stationary/constant graph mode.
    order = np.argsort(eigenvalues)[::-1]

    nontrivial_indices = order[1:]

    if len(nontrivial_indices) == 0:
        return np.zeros(len(X), dtype=float)

    selected_values = eigenvalues[nontrivial_indices]
    selected_vectors = eigenvectors[:, nontrivial_indices]

    diffusion_coordinates = (
        selected_vectors
        * (selected_values ** diffusion_time)[None, :]
    )

    # Weighted neighbour centroid in diffusion space.
    W, _ = _build_gaussian_graph(X)

    total_weight = W.sum(axis=1)

    neighbour_centroid = (
        W @ diffusion_coordinates
    ) / total_weight[:, None]

    # Distance from each transaction to its graph-neighbour centroid.
    isolation = np.sqrt(
        np.sum(
            (diffusion_coordinates - neighbour_centroid) ** 2,
            axis=1,
        )
    )

    return isolation


def _calculate_reconstruction_residual(X, W):
    """
    Prototype definition of R:
    distance between a transaction's features and the
    weighted average of its neighbours' features.
    """

    degree = W.sum(axis=1)

    reconstruction = (
        W @ X
    ) / degree[:, None]

    residual = np.sqrt(
        np.sum(
            (X - reconstruction) ** 2,
            axis=1,
        )
    )

    return residual


def detect_anomalies(
    df,
    date_column,
    amount_column,
    scales=6,
    epsilon=1.0,
    base_scale=10.5,
    omega0=6.0,
    beta=1.0,
    alpha=1.0,
    beta_weight=1.0,
    gamma=1.0,
    diffusion_time=1,
):
    """
    Detect unusual transactions using a graph-spectral prototype.

    Parameters
    ----------
    df : pandas.DataFrame
        Input voucher transactions.
    date_column : str
        Name of the date column.
    amount_column : str
        Name of the amount column.
    scales : int
        Number of graph-wavelet scales.
    epsilon : float
        Gaussian graph bandwidth parameter.
    base_scale : float
        Centre of the logarithmic scale range.
    omega0 : float
        Morlet-type central frequency parameter.
    beta : float
        Morlet filter bandwidth parameter.
    alpha : float
        Weight for E.
    beta_weight : float
        Weight for Q.
    gamma : float
        Weight for R.
    diffusion_time : int
        Diffusion coordinate time.

    Returns
    -------
    pandas.DataFrame
        Original transactions plus component scores and ranking.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(
            "The uploaded CSV contains no transactions."
        )

    if date_column not in df.columns:
        raise ValueError("Selected date column does not exist.")

    if amount_column not in df.columns:
        raise ValueError("Selected amount column does not exist.")

    if len(df) < 3:
        raise ValueError(
            "At least three transactions are required."
        )

    weights = np.array(
        [alpha, beta_weight, gamma],
        dtype=float,
    )

    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise ValueError(
            "alpha, beta_weight and gamma must be finite "
            "non-negative values."
        )

    if weights.sum() <= 0:
        raise ValueError(
            "At least one component weight must be positive."
        )

    # 1. Prepare features.
    result, X = _prepare_features(
        df,
        date_column,
        amount_column,
    )

    # 2. Build the Gaussian all-pairs graph.
    W, distances_squared = _build_gaussian_graph(
        X,
        epsilon=epsilon,
    )

    # 3. Spectral decomposition.
    (
        P,
        S,
        L,
        degree,
        eigenvalues,
        eigenvectors,
        frequencies,
    ) = _spectral_decomposition(W)

    # 4. Calculate E: wavelet energy.
    raw_E, scale_values, energy_by_scale = (
        _calculate_wavelet_energy(
            X,
            eigenvectors,
            frequencies,
            scales=scales,
            omega0=omega0,
            beta=beta,
            base_scale=base_scale,
        )
    )

    # 5. Calculate Q: diffusion-space isolation.
    # The same Gaussian graph bandwidth must be used here.
    order = np.argsort(eigenvalues)[::-1]
    nontrivial_indices = order[1:]

    if len(nontrivial_indices):
        selected_values = eigenvalues[nontrivial_indices]
        selected_vectors = eigenvectors[:, nontrivial_indices]

        diffusion_coordinates = (
            selected_vectors
            * (selected_values ** diffusion_time)[None, :]
        )

        neighbour_centroid = (
            W @ diffusion_coordinates
        ) / degree[:, None]

        raw_Q = np.sqrt(
            np.sum(
                (
                    diffusion_coordinates
                    - neighbour_centroid
                ) ** 2,
                axis=1,
            )
        )
    else:
        raw_Q = np.zeros(len(result), dtype=float)

    # 6. Calculate R: feature reconstruction residual.
    raw_R = _calculate_reconstruction_residual(X, W)

    # 7. Normalize the components independently.
    E = _minmax(raw_E)
    Q = _minmax(raw_Q)
    R = _minmax(raw_R)

    # 8. Combine E, Q and R.
    total_weight = alpha + beta_weight + gamma

    anomaly_score = (
        alpha * E
        + beta_weight * Q
        + gamma * R
    ) / total_weight

    # 9. Relative percentile within this dataset.
    anomaly_percentile = (
        pd.Series(anomaly_score)
        .rank(method="average", pct=True)
        .to_numpy()
        * 100.0
    )

    # 10. Add all results to the output.
    result["wavelet_energy_E"] = E
    result["geometric_isolation_Q"] = Q
    result["reconstruction_residual_R"] = R
    result["anomaly_score"] = anomaly_score
    result["anomaly_percentile"] = anomaly_percentile
    result["audit_review_flag"] = anomaly_percentile >= 95.0

    result = result.sort_values(
        "anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    return result
