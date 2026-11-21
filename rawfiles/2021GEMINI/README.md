# 2021GEMINI Validation Sample

This sequence contains 36 original ORF files, from `_C150295.ORF` through
`_C150330.ORF`.

## Visually Confirmed Meteors

All original frames were visually inspected. The following three contain meteors.
All unlisted original frames were visually classified as containing no meteor.

These labels describe the original exposures. A negative exposure can still
produce a disappearance residual in an adjacent-frame difference.

| File | Meteor |
|------|--------|
| `_C150303.ORF` | Yes |
| `_C150319.ORF` | Yes |
| `_C150326.ORF` | Yes |

## Verify the Files

`SHA256SUM` records checksums for all 36 ORF files. Run the appropriate command
from the repository root.

```bash
# macOS
(cd rawfiles/2021GEMINI && shasum -a 256 -c SHA256SUM)

# Linux
(cd rawfiles/2021GEMINI && sha256sum -c SHA256SUM)
```
