# Hotspot Stability P1.5 Plan

Goal: harden the topic-first hotspot layer so live AkShare/EastMoney instability does not silently produce empty results. Keep changes focused and backward-compatible.

Context:
- Current hotspot layer lives in `alphasift/hotspot.py` and CLI wiring is in `alphasift/cli.py`.
- Existing commands: `hotspots`, `hotspot <topic>`, `hotspot-cache`.
- Existing tests: `tests/test_hotspot.py`, `tests/test_cli.py`.
- Live AkShare/EastMoney endpoints may disconnect; tests must not use live network.

Required behavior:

1. Add degradation/source metadata to hotspot outputs
- Extend `HotspotSummary` and/or `HotspotDetail` minimally with optional fields that let CLI/JSON explain stability state:
  - `provider_used` or equivalent
  - `fallback_used` boolean
  - `source_errors` list[str]
  - `stale` boolean and/or `stale_age_hours` if practical
- Preserve existing fields and output compatibility. Adding fields is OK.
- Do not make normal no-provider tests noisy.

2. Add last-good cache fallback
- `discover_hotspots(...)` should accept an optional cache input, preferably `fallback_cache_path` or reuse `cache_path` if naming is cleaner.
- On live provider errors or empty live results, if a valid last-good cache exists, return those cached hotspots with fallback metadata set.
- Cache format should be compatible with `save_hotspots_json` output, but loader should tolerate older/simple JSON arrays and malformed rows.
- If no cache exists, still return [] as today, but include source errors where the return type allows it.

3. Harden `hotspot-cache`
- Add CLI option such as `--fallback-cache` or `--use-last-good` if needed, but keep existing command working.
- When live provider returns non-empty rows, write normal output/history/meta.
- When live provider fails/returns empty and a previous cache exists, do NOT overwrite the good cache with an empty list unless the user explicitly asks. Prefer:
  - read last-good cache;
  - write metadata indicating fallback/stale;
  - append history only if the fallback records are valid and clearly marked, or skip append to avoid polluting trend history. Choose a safe default and test it.
- Metadata should include provider, rows, fallback_used, source_errors, and generated_at.

4. Harden `hotspot <topic>` detail fallback
- If live provider fails to load board/constituents, and a fallback cache has a matching topic, return a summary from cache plus an empty stock list or cached leaders if available.
- If timeline path is provided and valid, timeline should still load even when provider fails.
- CLI should clearly show fallback/source error in `--explain` output.

5. Provider chain support
- Allow provider string with comma-separated providers, e.g. `--provider akshare,none` or `provider='akshare,none'`.
- Implement `none` as a no-network fallback returning no live rows.
- If unknown provider is present, record a source error instead of crashing.
- Keep injected fake provider object behavior for tests.

6. DSA runtime readiness helper, low risk
- Add a small helper and/or CLI-safe diagnostic if low risk: something like `alphasift.dsa.check_dsa_readiness(api_url)` returning structured status for missing URL / 401 / 404 / route present. Do not change existing DSA analysis behavior unless tests cover it.
- Add deterministic unit tests with monkeypatched requests or local fake response; no live network.
- If this feels too much, skip code changes and only document DSA operational caveat in plan; hotspot stability is higher priority.

7. Tests first / deterministic tests
Add tests before implementation for at least:
- loading last-good hotspot cache from `save_hotspots_json` shape;
- provider exception/empty live results falling back to cache;
- `hotspot-cache` does not overwrite a non-empty cache with empty provider results;
- CLI explain output includes fallback/source error when fallback is used;
- comma-separated provider string tolerates unknown provider and none;
- optional DSA readiness helper if implemented.

8. Verification commands
Run:
- `pytest tests/test_hotspot.py tests/test_cli.py tests/test_dsa.py tests/test_pipeline_dsa.py tests/test_dsa_adapter.py -q`
- `env -u LITELLM_MODEL -u LITELLM_API_KEY -u LITELLM_BASE_URL -u OPENAI_API_KEY -u OPENAI_BASE_URL -u GEMINI_API_KEY -u LLM_CHANNELS pytest -q`
- `python -m ruff check alphasift tests`
- `python -m alphasift.cli hotspots --provider none --explain`
- A smoke test using a temp existing cache and `hotspot-cache --provider none` proving it does not destroy the cache.

Constraints:
- No live network in tests.
- No secrets in code, tests, or docs.
- Keep existing commands backward-compatible.
- Keep changes focused; do not commit.
