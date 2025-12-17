# Exception and Error-Handling Catalog

This document summarizes the current exception surface, proposes `outputs`-specific
error types, and captures handling/logging conventions for the meteor detection
pipeline.

## Existing Exceptions

### Common / Shared
- Configuration validation relies on `ValueError`, `TypeError`, and `ImportError`
  raised by plugin base classes when required attributes or dependencies are
  missing.
  - `DataclassInputLoader` validation: `ValueError` for missing `plugin_name`,
    `TypeError` when `ConfigType` is not a dataclass or the instance type
    mismatches.【F:meteor_core/inputs/base.py†L189-L210】
  - `PydanticInputLoader` validation: `ImportError` when pydantic is missing and
    `TypeError` for non-pydantic configs or type mismatches.【F:meteor_core/inputs/base.py†L229-L252】
  - `DataclassOutputHandler` / `PydanticOutputHandler` mirror the same
    `ValueError` / `TypeError` / `ImportError` checks during initialization for
    output handlers.【F:meteor_core/outputs/base.py†L220-L291】
- Registry utilities emit `TypeError` and `AttributeError` when default output
  handler configs cannot be coerced or required path fields are absent, guarding
  against partially-constructed handlers.【F:meteor_core/outputs/registry.py†L135-L189】

### `inputs`-specific
- `RawLoaderConfig.__post_init__` raises `ValueError` when unsupported binning
  factors are supplied, preventing invalid RAW processing settings.【F:meteor_core/inputs/raw.py†L29-L40】
- `RawImageLoader.load` surfaces loader/runtime failures as generic `Exception`
  (propagated from RAW decoding helpers).【F:meteor_core/inputs/raw.py†L72-L91】
- Pipeline consumption paths catch loader errors while validating files,
  batch-processing images, and estimating thresholds to keep the run alive,
  returning error objects or fallback results instead of bubbling the exception
  upward.【F:meteor_core/pipeline.py†L330-L447】

## Proposed `outputs` Exceptions

Introduce output-focused error classes to clarify failure modes and simplify
handling:

| Exception | Responsibility | Typical Trigger |
| --- | --- | --- |
| `OutputWriteError` | Wraps filesystem or serialization failures when saving candidate files or debug artifacts. | Disk full, permission errors, corrupted image buffer. |
| `OutputValidationError` | Signals invalid output configuration or arguments detected at write time (e.g., missing directories, unsafe overwrite choices) beyond the existing type checks. | Derived paths not writable, absent debug folder, filename collisions when overwrite is disallowed. |

These classes would be raised by `BaseOutputHandler` implementations (e.g.,
`FileOutputHandler.save_candidate` / `save_debug_image`) and can carry context
about the target path and originating plugin.

## Handling Patterns and Logging Rules

| Handling pattern | Log level | Message guidance |
| --- | --- | --- |
| Re-throw (escalate to caller) | `ERROR` | Include plugin name, operation, and path; re-raise the original exception as cause. |
| Retry with backoff | `WARNING` on each attempt; `ERROR` after max retries | Log attempt number and waiting period; final log records retry exhaustion and surfaces the terminal exception. |
| Fallback (alternate handler/config) | `INFO` when switching; `WARNING` if triggered by recoverable error | Note original handler and fallback chosen; capture reason (exception class/message) in structured fields. |
| Failure counting / metrics only | `DEBUG` for individual increments; `INFO` when thresholds are crossed | Avoid noisy stack traces; log summary counts and the category of exceptions aggregated. |

## Propagation and Call-Site Impact

Current pipeline behavior swallows many loader/detector errors to keep batches
progressing, while output writes are executed without dedicated exception
boundaries:

- `_process_parallel` and `_process_sequential` call `save_candidate` directly;
  any raised `IOError`/`OSError` would currently unwind the run. Introducing
  `OutputWriteError`/`OutputValidationError` enables targeted handling (e.g.,
  retries or fallback output handlers) without masking unrelated bugs.【F:meteor_core/pipeline.py†L1158-L1175】【F:meteor_core/pipeline.py†L1220-L1235】
- Earlier stages already capture loader failures during validation and sampling;
  propagation remains limited to error objects in the result tuple.【F:meteor_core/pipeline.py†L330-L403】

**Callers likely to change when adopting output-specific exceptions**
- `MeteorDetectionPipeline._process_parallel` and `_process_sequential`: wrap
  `save_candidate`/`save_debug_image` invocations to map `OutputWriteError` into
  retry/fallback flows and to record failure counts alongside progress metrics.☆
- `process_image_batch`: if debug-image generation surfaces new output errors,
  convert them into batch-level warnings instead of halting the batch.☆
- Any CLI entrypoints (e.g., `detect_meteors_cli.py` or wrappers) that rely on
  pipeline return codes should optionally translate unrecoverable output errors
  into exit statuses or user-facing guidance.☆

☆ = requires code changes in callers to add explicit exception handling once the
new classes are introduced.
