# Topic-first Hotspots Implementation Plan

> **For Hermes:** Use Codex CLI to implement this plan in the alphasift git repository.

**Goal:** Add a topic-first “热点情报层” so alphasift can discover hot concepts first, drill into concept leaders, persist history/timeline, and later feed theme/capital strategies.

**Architecture:** Add a new `alphasift.hotspot` module that models hotspot boards, constituent stocks, roles, history, and timeline events. Wire two CLI commands: `alphasift hotspots` for ranked concept/industry discovery and `alphasift hotspot <topic>` for detail. Reuse AkShare/东方财富 board endpoints and the existing industry-cache board heat history format wherever possible.

**Tech Stack:** Python 3.10, pandas, argparse CLI, JSON/JSONL/CSV persistence, pytest. Keep network providers optional and make core ranking/test logic deterministic with injected DataFrames/mocks.

---

## Task 1: Add hotspot domain model and pure scoring helpers

**Objective:** Create deterministic hotspot ranking primitives independent of AkShare network calls.

**Files:**
- Create: `alphasift/hotspot.py`
- Create/modify: `tests/test_hotspot.py`

**Implementation notes:**
- Add dataclasses or plain dict builders for:
  - hotspot summary: topic/name, source, rank, change_pct, heat_score, trend_score, persistence_score, cooling_score, observations, state, sample_stock_count, leaders.
  - hotspot stock: code, name, change_pct, amount, turnover_rate, volume_ratio, net_inflow, is_limit_up, active_days, evidence_count, role, hot_stock_score.
  - timeline event: date, source, title, event_type, impact_score, related_codes.
- Implement pure helpers:
  - `compute_hotspot_heat_score(change_pct, rank)` compatible with existing `_board_heat_score` semantics.
  - `classify_hotspot_stage(state/trend/cooling/persistence/latest)` -> 初次异动 / 确认扩散 / 加速主升 / 分歧放量 / 降温退潮.
  - `score_hotspot_stock(row)` using涨幅、成交额、换手、量比、主力净流入、涨停、连续活跃、证据数.
  - `assign_stock_roles(scored_rows)` -> 核心龙头、助攻、补涨、后排、掉队.

## Task 2: Build provider/cache loaders

**Objective:** Load concept/industry boards, constituents, board heat history, and timeline JSONL from files or AkShare.

**Files:**
- Modify: `alphasift/hotspot.py`
- Create/modify: `tests/test_hotspot.py`

**Implementation notes:**
- Provide `discover_hotspots(provider='akshare', max_boards=80, history_path=None, top=20)`.
- Use `ak.stock_board_concept_name_em` and `ak.stock_board_industry_name_em` for board lists; normalize common Chinese columns.
- For each board, include rank/change_pct/heat_score; if history exists, merge `load_board_heat_trends` output.
- Provide `load_hotspot_history(path)` and `append_hotspot_history(path, hotspots, generated_at)` JSONL helpers.
- Provide timeline helpers: `load_hotspot_timeline(path, topic=None)` sorted by date, validate fields, skip malformed lines.

## Task 3: Build topic detail and hot-stock ranking

**Objective:** Support `hotspot <topic>` detail view with constituent stock ranking and timeline.

**Files:**
- Modify: `alphasift/hotspot.py`
- Create/modify: `tests/test_hotspot.py`

**Implementation notes:**
- Provide `get_hotspot_detail(topic, provider='akshare', top_stocks=10, timeline_path=None, history_path=None)`.
- Fetch concept constituents first (`stock_board_concept_cons_em(symbol=topic)`), fallback to industry constituents.
- Normalize columns: 代码/name/名称, 涨跌幅/change_pct, 成交额/amount, 换手率/turnover_rate, 量比/volume_ratio, 主力净流入/net_inflow.
- Add role assignment and stage/timeline summary.
- Do not call LLM here; code owns collection and ordering.

## Task 4: Add CLI commands and output formatters

**Objective:** Expose topic discovery and detail from the command line.

**Files:**
- Modify: `alphasift/cli.py`
- Create/modify: `tests/test_cli.py`

**CLI:**
- `alphasift hotspots --top 20 --explain --provider akshare --max-boards 80 --history-path data/hotspot.history.jsonl --output data/hotspots.json`
- `alphasift hotspot MLCC --top-stocks 10 --timeline --timeline-path data/hotspot_timeline.jsonl --explain`
- Add JSON output by default, compact text for `--explain`.
- Add `hotspot-cache` if needed to persist current hotspot ranking and history; keep `industry-cache` unchanged.

## Task 5: Integrate minimally with existing scoring context

**Objective:** Make generated hotspot cache reusable by existing candidate enrichment without broad strategy rewrites.

**Files:**
- Modify if low risk: `alphasift/industry.py`, `alphasift/scorer.py`, `alphasift/context.py` or docs/tests.

**Implementation notes:**
- Prefer low-risk P0/P1: output hotspot cache in a shape that can be supplied as an industry/concepts map or as future candidate context.
- If adding new fields to scoring, keep backwards compatibility and tests.

## Task 6: Verification

**Commands:**
- `pytest tests/test_hotspot.py tests/test_cli.py tests/test_industry.py tests/test_scorer.py -q`
- `pytest -q`
- `python -m alphasift.cli hotspots --provider none --explain` should not crash or should produce a clear unsupported/no-data message.
- If AkShare/network is available, smoke test with `python -m alphasift.cli hotspots --top 5 --explain`.

## Acceptance Criteria

- `alphasift hotspots --top N --explain` reports topic, rank, change, heat, trend, persistence, cooling, state/stage, sample count, and leaders.
- `alphasift hotspot <topic> --top-stocks N --timeline --explain` reports stage, hot-stock roles, and sorted timeline events.
- History and timeline are code-collected/sorted JSONL, not LLM inferred.
- Tests cover pure scoring, role assignment, history trend merge, malformed timeline skipping, and CLI formatting.
- Existing tests remain green.
