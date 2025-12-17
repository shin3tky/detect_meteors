# Logging plan for inputs and outputs

## 1. Inputs logging checklist
- **Leveling**
  - INFO: loader lifecycle milestones (discover -> open -> decode -> normalize), include target path and frame scope.
  - WARNING: recoverable fallbacks (missing EXIF, coerced dtype, resized frames).
  - ERROR: unrecoverable read/parse failures; emit before raising.
- **Message style**
  - Use lowercase, dot-separated event names (`inputs.open.start`, `inputs.decode.error`).
  - First field is a compact summary; follow with `key=value` pairs separated by spaces.
  - Keep messages single-line; defer multiline blobs to referenced files.
- **Structured keys** (always present where applicable)
  - `path` (input file), `loader` (plugin name), `frame` (index or `all`), `mode` (pipeline mode), `reason` for warnings/errors.
  - `duration_ms` for completed steps; `retry` count when applicable.
- **Privacy/size guardrails**
  - No raw pixel dumps or PII; truncate paths to absolute canonical form; cap value length at 256 chars.
- **Correlation**
  - Include `run_id` (pipeline invocation) and `trace_id` (per-file) when available to make upstream/downstream joins easy.

## 2. Outputs context requirements
Log records on the outputs side should attach:
- `output_path`: final file location (candidate or debug image).
- `source_path`: originating input path for traceability.
- `frame`: frame index or range used to generate the output.
- `mode`: processing mode (batch/cli vs api, dry-run flag).
- `handler`: output handler/plugin name.
- `overwrite`: boolean indicating whether existing files were replaced.
- `bytes_written` and `duration_ms` for completed writes.
- `progress_file` when progress state is updated.

## 3. Draft log templates and review notes
- `"inputs.open.start path=%s loader=%s frame=%s run_id=%s"`
- `"inputs.open.ok path=%s loader=%s frame=%s duration_ms=%d"`
- `"inputs.decode.error path=%s loader=%s frame=%s reason=%s"`
- `"outputs.write.start handler=%s output_path=%s source_path=%s frame=%s"`
- `"outputs.write.ok handler=%s output_path=%s bytes_written=%d duration_ms=%d overwrite=%s"`
- `"outputs.write_failed path=%s error=%s handler=%s source_path=%s frame=%s"`
- `"outputs.progress.saved progress_file=%s run_id=%s"`

Review notes:
- Keep placeholders aligned with the structured keys above; prefer `%s` for strings and `%d` for integers.
- Ensure the log emitter populates every placeholder; missing fields should default to sentinel values (`unknown`, `-1`).
- Document each template alongside the emitting function for discoverability.

## 4. Important events and deltas from current coverage
- Start: `inputs.open.start`, `outputs.write.start` — **new**; current code lacks explicit start markers, so downstream cannot measure durations.
- Success: `inputs.open.ok`, `outputs.write.ok` — **refined** to include durations and byte counts instead of generic success prints.
- Failure: `inputs.decode.error`, `outputs.write_failed` — **normalized** naming and required `reason/error` keys to replace ad-hoc exceptions.
- Retry: `inputs.open.retry` / `outputs.write.retry` (mirror start format, add `retry=%d`) — **new**; no retry visibility exists today.
- Progress checkpoint: `outputs.progress.saved` — **new**; current progress saves are silent, making resume debugging harder.
