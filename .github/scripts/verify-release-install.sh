#!/usr/bin/env bash
set -euo pipefail

release_tag=$1
dist=$(cd -- "$2" && pwd -P)
wheel="detect_meteors-$release_tag-py3-none-any.whl"
sdist="detect_meteors-$release_tag.tar.gz"

install_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/release-install.XXXXXX")
cd "$install_dir"

python_versions=(3.12 3.13)
supports_python_314=$(uv run --isolated --no-project --with 'packaging==26.3' --python 3.14 python - "$dist/$sdist" <<'PY'
from email.parser import BytesHeaderParser
from pathlib import Path
import platform
import sys
import tarfile

from packaging.specifiers import SpecifierSet

path = Path(sys.argv[1])
with tarfile.open(path, "r:gz") as archive:
    with archive.extractfile(f"{path.name.removesuffix('.tar.gz')}/PKG-INFO") as metadata:
        requires_python = BytesHeaderParser().parse(metadata)["Requires-Python"]

print("yes" if platform.python_version() in SpecifierSet(requires_python or "") else "no")
PY
)
if [[ "$supports_python_314" == yes ]]; then
  python_versions+=(3.14)
else
  echo "Skipping Python 3.14 because it is outside the release's Requires-Python range."
fi

for python_version in "${python_versions[@]}"; do
  for asset in "$wheel" "$sdist"; do
    echo "Checking $asset on Python $python_version"
    venv="$install_dir/$python_version-$asset"
    uv venv --python "$python_version" "$venv"
    uv pip install --python "$venv/bin/python" "$dist/$asset"
    uv pip check --python "$venv/bin/python"

    package_version=$("$venv/bin/python" -I -c 'from importlib.metadata import version; print(version("detect-meteors"))')
    cli_version=$("$venv/bin/detect-meteors" --version)
    if [[ "$package_version" != "$release_tag" || "$cli_version" != *" $release_tag" ]]; then
      echo "::error::Expected package and CLI version $release_tag. Package: $package_version. CLI: $cli_version."
      exit 1
    fi
    echo "$cli_version"
    "$venv/bin/detect-meteors" --help
  done
done

cd "$dist"
sha256sum --check SHA256SUMS

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf -v versions '%s, ' "${python_versions[@]}"
  printf 'python_versions=%s\n' "${versions%, }" >> "$GITHUB_OUTPUT"
fi
