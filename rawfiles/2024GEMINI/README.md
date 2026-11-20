# 2024GEMINI Validation Sample

This sequence contains 328 original ORF files, from `_C140200.ORF` through
`_C140527.ORF`.

## Visually Confirmed Meteors

All original frames were visually inspected. The following 12 contain meteors.
All unlisted original frames were visually classified as containing no meteor.

| File | Meteor |
|------|--------|
| `_C140215.ORF` | Yes |
| `_C140259.ORF` | Yes |
| `_C140275.ORF` | Yes |
| `_C140306.ORF` | Yes |
| `_C140338.ORF` | Yes |
| `_C140344.ORF` | Yes |
| `_C140359.ORF` | Yes |
| `_C140399.ORF` | Yes |
| `_C140403.ORF` | Yes |
| `_C140426.ORF` | Yes |
| `_C140430.ORF` | Yes |
| `_C140521.ORF` | Yes |

These labels describe the original exposures. A negative exposure can still
produce a disappearance residual in an adjacent-frame difference.

## Verify the Files

`SHA256SUM` records checksums for all 328 ORF files. Run the appropriate command
from the repository root.

```bash
# macOS
(cd rawfiles/2024GEMINI && shasum -a 256 -c SHA256SUM)

# Linux
(cd rawfiles/2024GEMINI && sha256sum -c SHA256SUM)
```
