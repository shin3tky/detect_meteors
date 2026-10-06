# 2024GEMINI_AIRCRAFT Validation Sample

This directory is tracked in Git so a repository clone can reproduce the
aircraft trail validation. It contains 12 original ORF files, approximately
229 MB in total. The RAW images are excluded from the Python wheel and source
distribution; use the repository checkout for the complete dataset.

## Owner-Confirmed Contents

All 12 images contain aircraft trails. Only `_C140338.ORF` and `_C140344.ORF`
contain owner-confirmed meteors.

| File | Aircraft | Meteor | Detection role |
|------|----------|--------|----------------|
| `_C140335.ORF` | Yes | No | Initial baseline; not scored |
| `_C140336.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140337.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140338.ORF` | Yes | Yes | Current frame in adjacent-pair detection |
| `_C140339.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140340.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140341.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140342.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140343.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140344.ORF` | Yes | Yes | Current frame in adjacent-pair detection |
| `_C140345.ORF` | Yes | No | Current frame in adjacent-pair detection |
| `_C140346.ORF` | Yes | No | Current frame in adjacent-pair detection |

These labels describe the original images, not the difference masks. Absolute
frame differences can retain an object's disappearance in the next pair.

## Verify the Files

From the repository root, verify the checksums:

```bash
# macOS
(cd rawfiles/2024GEMINI_AIRCRAFT && shasum -a 256 -c SHA256SUMS)

# Linux
(cd rawfiles/2024GEMINI_AIRCRAFT && sha256sum -c SHA256SUMS)
```

## Reproduce the Validation

From the repository root:

```bash
uv sync
uv run python detect_meteors_cli.py \
  --config config_examples/aircraft_trail_sample.yaml \
  --no-roi --no-resume --debug-image --profile
```

The sample uses two workers; add `--workers 1` on a single-core machine.
There are 11 analyzed pairs. Both meteor images should remain candidates;
aircraft likelihood does not exclude mixed images.

See the [usage guide](../../docs/aircraft_light_trails_hook_design.md) and
[validation report](../../docs/aircraft_sample_validation.md) for the settings,
output locations, and measured results.
