# FAIR Subgroup Weight Balancing: Worked Example

A minimal worked example of the combined NSDUH + FAIR subgroup weight formula:

```
w_i = s_i * min_k(n_hat_k) / n_hat_{k(i)}
```

where `s_i` is the NSDUH person-level analysis weight and `n_hat_k` is the sum of survey weights within FAIR subgroup `k`. (The `/Y` rescaling for `n_pool_years` is omitted here for clarity; it applies uniformly when pooling multiple survey years.)

Subgroup in this example: **Sex** (Male, Female).

## Step 1: Raw inputs

| i | Sex (k) | s_i (NSDUH weight) |
|---|---------|--------------------|
| 1 | Male    | 1,000 |
| 2 | Male    | 1,500 |
| 3 | Male    | 2,500 |
| 4 | Female  |   800 |
| 5 | Female  | 1,200 |
| 6 | Female  | 1,000 |

## Step 2: Compute subgroup totals n̂_k

| Subgroup k | n̂_k = Σ s_i |
|------------|--------------|
| Male       | 1,000 + 1,500 + 2,500 = **5,000** |
| Female     |   800 + 1,200 + 1,000 = **3,000** |

Males carry 5,000 / 8,000 ≈ 62.5% of the total weight, females only 37.5%. A model trained on this will favor accuracy on the male subgroup.

## Step 3: Pick the target and compute correction factors

`target = min_k(n̂_k) = min(5,000, 3,000) = 3,000`

| Subgroup k | Correction = target / n̂_k |
|------------|----------------------------|
| Male       | 3,000 / 5,000 = **0.60** |
| Female     | 3,000 / 3,000 = **1.00** |

Females are the smaller subgroup, so they are left alone. Males get scaled down to match.

## Step 4: Apply w_i = s_i · target / n̂_{k(i)}

| i | Sex | s_i   | Correction | Final w_i |
|---|-----|-------|------------|-----------|
| 1 | M   | 1,000 | 0.60       | **600**   |
| 2 | M   | 1,500 | 0.60       | **900**   |
| 3 | M   | 2,500 | 0.60       | **1,500** |
| 4 | F   |   800 | 1.00       | **800**   |
| 5 | F   | 1,200 | 1.00       | **1,200** |
| 6 | F   | 1,000 | 1.00       | **1,000** |

## Step 5: Verify totals are balanced

| Subgroup | Total weight before | Total weight after |
|----------|---------------------|--------------------|
| Male     | 5,000               | **3,000**          |
| Female   | 3,000               | **3,000**          |

Both subgroups now contribute equally to the training loss.

## Summary

Scale every subgroup down to match the smallest subgroup's total weight, so each subgroup contributes equally to the loss while each respondent's relative weight within their subgroup is unchanged.

### What is preserved

- **Within-subgroup relative sampling weights.** A single constant factor is applied per subgroup, so the NSDUH survey design within each subgroup is untouched.
- **No inflation.** Because the target is `min_k(n̂_k)`, no respondent's final weight ever exceeds their original NSDUH weight.

### What changes

- **Aggregate loss contribution per subgroup.** Equalized across FAIR subgroups (the C1 condition).
