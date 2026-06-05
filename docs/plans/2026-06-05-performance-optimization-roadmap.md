# AlphaSift Performance / Reliability Optimization Roadmap

Goal: after the hotspot fallback work, optimize AlphaSift as a practical quant-research screening engine. Borrow the useful patterns from mature platforms without turning AlphaSift into a full backtesting/trading framework.

References / borrowed patterns:
- Qlib: loose-coupled infrastructure + workflow + analyzer layers; cache expensive data retrieval; keep modules independently testable.
- Freqtrade: strategy optimization depends on repeatable cached historical data and clear deterministic tests; heavy optional workflows should not slow normal runs.
- Backtrader: analyzer/observer pattern; evaluation metrics should be explicit, persisted, and cheap to recompute.

Current verified baseline:
- Worktree already has uncommitted hotspot/CLI/DSA changes.
- Full test suite passes: 124 passed.
- First optimization implemented: candidate-level LLM context fetch is now parallelized with deterministic output ordering and per-row error isolation.

## P0 — Already implemented in this pass: parallel candidate context fetch

Files:
- `alphasift/candidate_context.py`
- `tests/test_candidate_context.py`

Why:
- LLM candidate enrichment previously fetched `candidates × providers` sequentially. For `news,fund_flow,announcement` this becomes the biggest wall-time cost before the LLM call.
- This matches the Qlib/Freqtrade style of separating slow data collection from deterministic downstream scoring.

Acceptance:
- Rows preserve original candidate order even when workers finish out of order.
- A failure for one candidate/provider records a row error and does not abort other candidates.
- Cache behavior remains stable.

Verification:
- `python -m pytest tests/test_candidate_context.py -q`
- `python -m pytest -q`

## P1 — Parallel daily K-line enrichment

Files:
- `alphasift/daily.py`
- `alphasift/evaluate.py`
- `tests/test_daily.py`
- `tests/test_pipeline_daily.py`
- `tests/test_store_evaluate.py`

Why:
- `DAILY_ENRICH_MAX_CANDIDATES=100` with retries can dominate runtime.
- Evaluation price-path fetching has the same shape: independent per-code network calls.

Plan:
1. Introduce bounded ThreadPoolExecutor in `enrich_daily_features` and `_fetch_price_paths`.
2. Preserve input index/order in merged output.
3. Keep per-code errors in `df.attrs["daily_errors"]` / evaluation degradation.
4. Add conservative default worker count, configurable later if needed.

Tests:
- `tests/test_daily.py::test_enrich_daily_features_fetches_rows_concurrently_preserving_index`
- `tests/test_daily.py::test_enrich_daily_features_keeps_successful_rows_when_one_fetch_fails`
- `tests/test_store_evaluate.py::test_evaluate_saved_runs_uses_parallel_price_path_fetch_without_order_drift`

## P2 — Daily history cache

Files:
- `alphasift/daily.py`
- `alphasift/config.py`
- `alphasift/evaluate.py`

Why:
- Screening and T+N evaluation repeatedly refetch the same `(code, source, lookback_days)` data.
- Freqtrade-style workflows assume historical data is cached before repeated optimization/evaluation.

Plan:
1. Add cache directory under `ALPHASIFT_DATA_DIR/daily_history`.
2. Key by normalized code + source + lookback days + optional trade-date/version.
3. Respect TTL and bypass cache on explicit refresh option later.
4. Mark stale/fallback cache in attrs/degradation when used after live failure.

Tests:
- `tests/test_daily.py::test_fetch_daily_history_uses_cache_until_ttl`
- `tests/test_daily.py::test_fetch_daily_history_refetches_after_cache_expiry`
- `tests/test_store_evaluate.py::test_evaluate_saved_runs_uses_cached_price_paths`

## P3 — Last-good snapshot cache

Files:
- `alphasift/snapshot.py`
- `alphasift/config.py`
- `tests/test_snapshot.py`

Why:
- Snapshot fallback currently only tries live providers and raises if all fail. Hotspot fallback has already shown the better pattern: use last-good data with explicit stale metadata.
- This improves weekend/holiday/offline reliability.

Plan:
1. Add optional `fallback_snapshot_path` or default `data/snapshot.last_good.json`.
2. On successful live fetch, save normalized snapshot + metadata.
3. On full live failure, load last-good cache and set attrs:
   - `snapshot_source="last_good_cache"`
   - `fallback_used=True`
   - `stale_age_hours`
   - `source_errors`
4. Never hide staleness in CLI/explain output.

Tests:
- `tests/test_snapshot.py::test_fetch_snapshot_with_fallback_uses_last_good_cache_after_all_sources_fail`
- `tests/test_snapshot.py::test_snapshot_fallback_marks_stale_source_metadata`

## P4 — Vectorize industry map application

Files:
- `alphasift/industry.py`
- `tests/test_industry.py`

Why:
- Full-market enrichment loops row-by-row with `.iterrows()` / `.at`. That is fine for tiny tests but avoidable for 5k+ A-share rows.

Plan:
1. Build a mapping DataFrame keyed by normalized code.
2. Merge once into snapshot.
3. Apply deterministic merge rules for concepts/summary/heat fields.
4. Keep existing provider/file behavior unchanged.

Tests:
- `tests/test_industry.py::test_enrich_industry_concepts_from_file`
- `tests/test_industry.py::test_enrich_industry_concepts_preserves_existing_concepts_after_vectorized_merge`

## P5 — Cache AkShare board provider output

Files:
- `alphasift/industry.py`
- `alphasift/hotspot.py`

Why:
- Provider mode can perform up to `2 * max_boards` constituent calls per refresh. Cache once, reuse in screen/hotspot flows.

Tests:
- `tests/test_industry.py::test_fetch_akshare_board_map_uses_provider_cache`
- `tests/test_industry.py::test_fetch_akshare_board_map_limits_and_merges_parallel_results`

## P6 — Metadata-first saved run listing

Files:
- `alphasift/store.py`
- `alphasift/evaluate.py`

Why:
- `list_saved_runs` parses full result JSONs, then evaluation loads them again. Backtrader-style analyzers should make summaries cheap.

Plan:
1. Save a small sidecar metadata file at screen-save time, or support a fast partial parse.
2. Filter by strategy before loading full run payload.
3. Keep backward compatibility when metadata is missing.

Tests:
- `tests/test_store_evaluate.py::test_list_saved_runs_reads_only_metadata_until_limit`
- `tests/test_store_evaluate.py::test_evaluate_saved_runs_filters_strategy_before_loading_runs`

## P7 — Cache strategy/config loading

Files:
- `alphasift/strategy.py`
- `alphasift/config.py`

Why:
- Every `screen()` reloads YAML and revalidates strategy directory sync. This is unnecessary in agent/daemon workflows.

Plan:
1. Cache `load_all_strategies` by directory path + strategy file mtimes.
2. Keep explicit invalidation simple: mtime change invalidates.
3. Avoid caching environment values that tests monkeypatch unless cache key includes relevant env state.

Tests:
- `tests/test_strategy.py::test_load_all_strategies_uses_cache_until_yaml_mtime_changes`
- `tests/test_config.py::test_config_from_env_loads_env_file_once_per_process`

## P8 — Bound LLM context earlier

Files:
- `alphasift/context.py`
- `alphasift/ranker.py`

Why:
- Prompt size drives LLM latency/cost. Current truncation is functional but late/blunt.

Plan:
1. Allocate token/char budget by section: market context, snapshot summary, candidate context, DSA context.
2. Preserve required candidate identity fields and top-ranked candidates first.
3. Emit degradation if low-priority context is trimmed.

Tests:
- `tests/test_ranker.py::test_ranking_prompt_is_bounded_and_keeps_required_fields`
- `tests/test_context.py::test_build_llm_context_preserves_candidate_sections_under_budget`

## P9 — One-pass hard filter mask

Files:
- `alphasift/filter.py`

Why:
- Repeated DataFrame filtering is simple but scans/mutates multiple times. A single mask is faster and easier to explain.

Tests:
- `tests/test_filter.py::test_apply_hard_filters_matches_one_pass_numeric_and_daily_filters`
- keep existing missing-field and empty-frame tests green.

## P10 — Shared normalization utilities

Files:
- create `alphasift/utils.py` or `alphasift/normalize.py`
- update `pipeline.py`, `industry.py`, `context.py`, `evaluate.py`, `ranker.py`, `post_analysis.py`

Why:
- Code normalization and safe parsing are duplicated. Drift here can silently mismatch candidates, cached rows, and evaluation results.

Tests:
- `tests/test_code_utils.py::test_normalize_code_accepts_numeric_suffixed_and_prefixed_codes`
- update existing normalization coverage in `test_ranker.py`, `test_store_evaluate.py`, `test_industry.py`.

Recommended next Codex task:

Implement P1 + P2 together only if time allows; otherwise do P1 first. Daily K-line parallelism provides immediate runtime improvement with low conceptual risk. Daily-history cache should follow once parallel behavior is stable.
