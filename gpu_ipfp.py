import cupy as cp


def gpu_ipfp(
    original,
    aggregates,
    dimensions,
    max_iter=2000,
    tol=1e-6,
    check_every=10,
    dtype=cp.float64,
    verbose=True,
    return_history=False
):
    """
    GPU-accelerated multidimensional Iterative Proportional Fitting.

    Parameters
    ----------
    original : array
        Initial N-dimensional distribution.

    aggregates : list of arrays
        Target marginal distributions.

    dimensions : list of tuples/lists
        Dimensions retained by each aggregate.

        Example:
            original.shape = (10, 20, 30)

            dimensions = [
                (0, 1),
                (1, 2),
                (0, 2)
            ]

            aggregates = [
                array(shape=(10, 20)),
                array(shape=(20, 30)),
                array(shape=(10, 30))
            ]

    max_iter : int
        Maximum number of complete IPFP iterations.

    tol : float
        Maximum permitted absolute marginal error.

    check_every : int
        Check convergence every N iterations.

    dtype : cupy dtype
        Floating-point precision.

    verbose : bool
        Print convergence information.

    return_history : bool
        Return convergence history if True.

    Returns
    -------
    X : cupy.ndarray
        Fitted multidimensional distribution.

    converged : bool
        Whether convergence criterion was reached.

    history : list, optional
        Convergence history if requested.
    """

    if len(aggregates) != len(dimensions):
        raise ValueError(
            "aggregates and dimensions must have equal lengths"
        )

    if not aggregates:
        raise ValueError("At least one constraint is required")

    if max_iter < 1 or check_every < 1:
        raise ValueError("max_iter and check_every must be positive")

    if tol < 0:
        raise ValueError("tol must be nonnegative")

    # Copy input onto GPU
    X = cp.array(original, dtype=dtype, copy=True)

    ndim = X.ndim
    all_axes = set(range(ndim))

    if not bool(cp.all(cp.isfinite(X)).item()):
        raise ValueError("Original array contains NaN or infinity")

    if bool(cp.any(X < 0).item()):
        raise ValueError("Original array contains negative values")

    # Prepare constraints
    constraints = []

    for target, dims in zip(aggregates, dimensions):

        dims = tuple(
            int(d) % ndim if -ndim <= int(d) < ndim else int(d)
            for d in dims
        )

        if len(set(dims)) != len(dims):
            raise ValueError("Duplicate dimensions in constraint")

        if any(d < 0 or d >= ndim for d in dims):
            raise ValueError("Invalid dimension index")

        target = cp.asarray(target, dtype=dtype)

        expected_shape = tuple(X.shape[d] for d in dims)

        if target.shape != expected_shape:
            raise ValueError(
                f"Expected aggregate shape {expected_shape}, "
                f"received {target.shape}"
            )

        if not bool(cp.all(cp.isfinite(target)).item()):
            raise ValueError("Constraint contains NaN or infinity")

        if bool(cp.any(target < 0).item()):
            raise ValueError("Constraint contains negative values")

        # Dimensions to sum over
        sum_axes = tuple(sorted(all_axes - set(dims)))

        # Reorder target axes to match original axis order
        order = sorted(range(len(dims)), key=lambda j: dims[j])

        if len(order) > 1:
            target = cp.transpose(target, order)

        # Broadcast target to original dimensionality
        broadcast_shape = tuple(
            X.shape[d] if d in dims else 1
            for d in range(ndim)
        )

        target = target.reshape(broadcast_shape)

        # Preallocate current marginal and adjustment factor
        current = cp.empty(broadcast_shape, dtype=dtype)
        factor = cp.empty(broadcast_shape, dtype=dtype)

        constraints.append({
            "target": target,
            "axes": sum_axes,
            "current": current,
            "factor": factor
        })

    # Check grand totals before fitting
    totals = [
        float(c["target"].sum().item())
        for c in constraints
    ]

    if not all(
        abs(t - totals[0]) <= max(tol, 1e-8) * max(1, totals[0])
        for t in totals
    ):
        raise ValueError(
            f"Constraint grand totals are inconsistent: {totals}"
        )

    history = []
    converged = False

    # Handle completely empty data
    if totals[0] == 0:
        X.fill(0)
        if return_history:
            return X, True, [(0, 0.0, 0.0)]
        return X, True

    # Main IPFP loop
    for iteration in range(1, max_iter + 1):

        for j, c in enumerate(constraints):

            target = c["target"]
            axes = c["axes"]
            current = c["current"]
            factor = c["factor"]

            # Compute current marginal distribution
            cp.sum(
                X,
                axis=axes,
                keepdims=True,
                out=current
            )

            # Detect impossible positive targets
            # with zero support
            impossible = (current == 0) & (target > 0)

            if bool(cp.any(impossible).item()):
                locations = cp.argwhere(impossible)

                print(f"\nError at iteration {iteration}")
                print(f"Constraint: {j}")
                print(f"Dimensions: {dimensions[j]}")
                print(f"Number of impossible cells: {len(locations)}")

                for idx in locations[:10].get():
                    idx = tuple(int(x) for x in idx)
                    print(
                        f"Index: {idx}",
                        f"Target: {float(target[idx])}",
                        f"Current: {float(current[idx])}"
                    )

                raise ValueError(
                    f"Constraint {j} has positive targets with zero support"
                )

            # Calculate proportional adjustment
            # Avoid division by zero
            cp.maximum(current, 1e-300, out=factor)

            # Calculate adjustment factors
            cp.divide(target, factor, out=factor)

            # Where both current and target are zero,
            # use a factor of 1
            cp.copyto(factor, 1.0, where=current == 0)

            # Apply adjustment in place
            cp.multiply(X, factor, out=X)

        # Check convergence periodically
        if iteration % check_every == 0 or iteration == max_iter:

            max_abs_error = 0.0
            max_rel_error = 0.0

            for c in constraints:

                cp.sum(
                    X,
                    axis=c["axes"],
                    keepdims=True,
                    out=c["current"]
                )

                diff = cp.abs(c["current"] - c["target"])

                abs_err = float(cp.max(diff).item())

                # Relative error, normalized by the
                # largest target value in this margin.
                scale = max(
                    float(cp.max(c["target"]).item()),
                    1.0
                )

                rel_err = abs_err / scale

                max_abs_error = max(max_abs_error, abs_err)
                max_rel_error = max(max_rel_error, rel_err)

            history.append((
                iteration,
                max_abs_error,
                max_rel_error
            ))

            if verbose:
                print(
                    f"Iteration {iteration}: "
                    f"absolute error = {max_abs_error:.8g}, "
                    f"relative error = {max_rel_error:.8g}"
                )

            if max_abs_error <= tol:
                converged = True

                if verbose:
                    print(f"Converged after {iteration} iterations")

                break

    if not converged and verbose:
        print(
            f"Did not converge after {max_iter} iterations"
        )

    if return_history:
        return X, converged, history

    return X, converged