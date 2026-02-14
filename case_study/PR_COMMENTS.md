# PR Review Comments for `utils.py`

## Critical Issues

### 1. `extend_promotions_days` (line 63) — `DataFrame.append()` removed in pandas 2.0

`DataFrame.append()` was deprecated in pandas 1.4 and **removed in pandas 2.0**. This function crashes with `AttributeError` on any modern pandas installation.

**Fix:** Replace the loop + `.append()` pattern with `pd.concat`:

```python
frames = [promotions.copy().assign(promotion_id=promotion_id)]
for days_to_add in range(1, n_days):
    frame = initial_promotions.copy().assign(promotion_id=promotion_id)
    frame.index += pd.Timedelta(days_to_add, "d")
    frames.append(frame)
return pd.concat(frames)
```

### 2. `clean` (line 29) — Data leakage via `bfill()`

`bfill()` fills missing values with the *next* non-null observation — a future value. In a forecasting context, this leaks future information into the training data. This is especially problematic for demand forecasting where missing values may cluster around supply disruptions or holidays.

**Recommendation:** Use `ffill()` (forward fill) or `interpolate(method='linear')` to avoid leakage. Fall back to the group mean only for leading NaNs that cannot be forward-filled.

```python
def clean(ts: pd.Series) -> pd.Series:
    return ts.interpolate(method="linear").bfill().fillna(ts.mean())
```

---

## High Priority

### 3. `clean_demand_per_group` (lines 33–37) — In-place mutation + O(n × groups) performance

Two issues here:

1. **In-place mutation**: The function modifies the input DataFrame directly via `.loc` assignment. Callers may not expect this side effect. Returning a modified copy would be safer.

2. **Performance**: The nested loop performs repeated full-DataFrame boolean indexing. For `g` groups and `n` rows, this is O(g × n) per iteration. A `groupby`-based approach is both clearer and faster:

```python
def clean_demand_per_group(demand: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["demand"] = demand.groupby(
        ["supermarket", "sku"]
    )["demand"].transform(clean)
    return demand
```

### 4. `merge` (lines 42–46) — `how="outer"` may create orphan rows

If a promotion date falls outside the demand date range, the outer join creates rows with `NaN` for `demand`, `sku`, and `supermarket`. This silently inflates the dataset. A `how="left"` join would be safer — promotions without matching demand dates are dropped rather than creating phantom rows.

**Minimal repro:**
```python
# If a promotion exists for 2022-01-01 but demand only goes to 2021-12-30,
# outer join creates a row with demand=NaN, sku=NaN, supermarket=NaN.
```

---

## Moderate Issues

### 5. `aggregate_to_weekly` (line 69) — `FutureWarning` in pandas 2.x

`grouped.apply(lambda df: ...)` includes the grouping columns (`sku`, `supermarket`) inside the lambda by default, which triggers a `FutureWarning`. Add `include_groups=False`:

```python
weekly = grouped.apply(
    lambda df: df.resample("W").agg({"demand": "sum", "promotion": "max"}),
    include_groups=False
)
```

### 6. `parse_time` (line 11) — Row-by-row parsing is slow

`datetime.strptime` is called row-by-row via `.apply()`. `pd.to_datetime` handles `"%Y-%m-%d"` natively and is vectorized — roughly 10–50x faster on large DataFrames:

```python
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
```

### 7. `merge` (line 83) — `fillna` on object dtype triggers `FutureWarning`

The `promotion` column after the outer join is of object dtype (mix of `True` and `NaN`). Calling `.fillna(False)` triggers a deprecation warning about silent downcasting. Use `np.where` instead:

```python
demand["promotion"] = np.where(demand["promotion"].isna(), False, demand["promotion"])
```

---

## Minor / Style

- **Missing type hints** on `parse_time`, `read_demand`, `read_promotions`, `merge`, `extend_promotions_days`, `aggregate_to_weekly`.
- **Module docstring** (lines 1–5) is generic. Consider describing the expected workflow: `read_demand → clean_demand_per_group → extend_promotions_days → merge → aggregate_to_weekly`.
- **Variable naming**: `sus` (line 33) is unclear — `supermarkets` would be more descriptive.
- **Shadow warning**: The lambda parameter in `read_demand` (line 15) shadows the outer `df` variable.

---

## Unexpected Behaviour in `merge`

The intended workflow is:
1. `extend_promotions_days(promos, 7)` to expand each promo start date into 7 daily rows
2. `merge(demand, extended_promos)` to flag promotion days in the demand DataFrame

However, since `extend_promotions_days` crashes on pandas 2.0+, if a user calls `merge()` directly with the raw 15-row promotions DataFrame, only the promo *start dates* are flagged. The remaining 6 days of each 7-day promotion window are missed entirely. The result is `promotion=True` for only 15 of ~9,855 rows instead of the expected ~105 rows (15 promos × 7 days).

**Suggestion:** Add a validation check in `merge()` or document the required call order.

---

## Suggested Tests

| Test | Purpose |
|------|---------|
| `test_clean_no_future_leakage` | Place a NaN at position `i`, verify the filled value does not use data from `i+1` onwards |
| `test_extend_promotions_days_length` | After extending by `n_days=7`, assert `len(result) == 7 * len(input)` |
| `test_extend_promotions_days_date_range` | For a promo starting on Jan 1, verify rows exist for Jan 1–Jan 7 |
| `test_merge_no_orphan_rows` | With promos inside the demand date range, assert merged shape equals demand shape |
| `test_merge_promotion_count` | After extending 15 promos by 7 days, assert `promotion.sum() == 105` |
| `test_clean_demand_per_group_no_nulls` | After cleaning, assert `demand["demand"].isna().sum() == 0` |
| `test_aggregate_to_weekly_shape` | For 1,095 days, expect ~157 weekly rows per group |
| `test_clean_demand_per_group_idempotent` | Calling twice should produce the same result |
