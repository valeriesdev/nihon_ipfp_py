
import cupy as cp

def diagnose_ipfp(result, aggregates, dimensions):
    all_axes = set(range(result.ndim))

    for j, (target, dims) in enumerate(zip(aggregates, dimensions)):
        dims = tuple(d % result.ndim for d in dims)
        sum_axes = tuple(sorted(all_axes - set(dims)))

        # Calculate margin
        fitted = result.sum(axis=sum_axes)

        # Match the target's original dimension order
        retained = sorted(dims)
        perm = [retained.index(d) for d in dims]
        fitted = cp.transpose(fitted, perm)

        target = cp.asarray(target)

        errors = cp.abs(fitted - target)

        max_error = float(errors.max().item())
        total_error = float(errors.sum().item())

        # Location of largest discrepancy
        worst = cp.unravel_index(
            cp.argmax(errors), errors.shape
        )
        worst = tuple(int(x) for x in worst)

        print(f"\nConstraint {j}: dimensions {dims}")
        print(f"  Max error:   {max_error:.6f}")
        print(f"  Total error: {total_error:.6f}")
        print(f"  Target sum:  {float(target.sum()):.6f}")
        print(f"  Fitted sum:  {float(fitted.sum()):.6f}")
        print(f"  Worst index: {worst}")
        print(f"  Target:      {float(target[worst]):.6f}")
        print(f"  Fitted:      {float(fitted[worst]):.6f}")

diagnose_ipfp(result, aggregates, dimensions)

I = cp.asarray(aggregates[0], dtype=cp.float64)
O = cp.asarray(aggregates[1], dtype=cp.float64)
P = cp.asarray(aggregates[2], dtype=cp.float64)

industry_totals = I.sum(axis=2)
occupation_totals = O.sum(axis=2)

diff = industry_totals - occupation_totals

print("Maximum difference:", float(cp.max(cp.abs(diff)).item()))
print("Total absolute difference:", float(cp.sum(cp.abs(diff)).item()))
print("Number of mismatches:",
      int(cp.count_nonzero(cp.abs(diff) > 1e-8).item()))

# Find largest mismatches
indices = cp.argwhere(cp.abs(diff) > 1e-8).get()

for s, n in indices[:20]:
    print(
        f"Sex {s}, neighborhood {n}:",
        f"industry={float(industry_totals[s, n]):.6f}",
        f"occupation={float(occupation_totals[s, n]):.6f}"
    )

# Investigate neighborhood 1212, sex 1
print("\nProblematic neighborhood:")
print("Industry total:", float(industry_totals[1, 1212].item()))
print("Occupation total:", float(occupation_totals[1, 1212].item()))


# Sum neighborhoods to obtain sex × industry
municipal_industry = I.sum(axis=1)

# Sum age and occupation in prefecture table
pref_industry = P.sum(axis=(1, 3))

# Sum neighborhoods to obtain sex × occupation
municipal_occupation = O.sum(axis=1)

# Sum age and industry in prefecture table
pref_occupation = P.sum(axis=(1, 2))

print("Industry vs prefecture:",
      float(cp.max(cp.abs(
          municipal_industry - pref_industry
      )).item()))

print("Occupation vs prefecture:",
      float(cp.max(cp.abs(
          municipal_occupation - pref_occupation
      )).item()))
