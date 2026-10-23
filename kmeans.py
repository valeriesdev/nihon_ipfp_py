import cupy as cp
import faiss
import faiss.contrib.torch_utils
import torch

def gpu_kmeans(n, d, k, niter, nredo, tol, X):
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    rng = cp.random.RandomState(42)

    best_inertia = float("inf")
    best_labels = None
    best_centroids = None

    for restart in range(nredo):
        selected = rng.choice(n, size=k, replace=False)
        centroids = cp.ascontiguousarray(X[selected])

        for iteration in range(niter):
            cp.cuda.get_current_stream().synchronize()

            index.reset()
            index.add_c(k, faiss.cast_integer_to_float_ptr(centroids.data.ptr))
            distances = cp.empty((n, 1), dtype=cp.float32)
            labels = cp.empty((n, 1), dtype=cp.int64)

            index.search_c(
                n,
                faiss.cast_integer_to_float_ptr(X.data.ptr),
                1,
                faiss.cast_integer_to_float_ptr(distances.data.ptr),
                faiss.cast_integer_to_idx_t_ptr(labels.data.ptr)
            )

            # Wait for FAISS GPU operations to finish before CuPy reads the results
            cp.cuda.Device().synchronize()
            labels = labels.ravel()

            assert index.ntotal == k
            assert bool(cp.all((labels >= 0) & (labels < k))), ("FAISS returned invalid cluster labels")

            sums = cp.zeros((k, d), dtype=cp.float32)
            cp.add.at(sums, labels, X)

            counts = cp.bincount(labels, minlength=k)

            new_centroids = sums / cp.maximum(
                counts[:, None], 1
            ).astype(cp.float32)

            empty = cp.where(counts == 0)[0]

            if empty.size:
                replacements = rng.choice(n, size=int(empty.size), replace=False)
                new_centroids[empty] = X[replacements]

            shift = float(cp.max(cp.abs(new_centroids - centroids)))
            centroids = cp.ascontiguousarray(new_centroids, dtype=cp.float32)
            if shift < tol:
                break

        cp.cuda.get_current_stream().synchronize()

        index.reset()
        index.add_c(
            k,
            faiss.cast_integer_to_float_ptr(centroids.data.ptr)
        )

        distances = cp.empty((n, 1), dtype=cp.float32)
        labels = cp.empty((n, 1), dtype=cp.int64)

        index.search_c(
            n,
            faiss.cast_integer_to_float_ptr(X.data.ptr),
            1,
            faiss.cast_integer_to_float_ptr(distances.data.ptr),
            faiss.cast_integer_to_idx_t_ptr(labels.data.ptr)
        )

        cp.cuda.Device().synchronize()
        labels = labels.ravel()
        assert bool(cp.all((labels >= 0) & (labels < k))), ("FAISS returned invalid cluster labels")
        inertia = float(cp.sum(distances))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
    return best_inertia, best_labels, best_centroids