#!/usr/bin/env bash
set -euo pipefail

release_tag=$1
dist=$2
wheel="detect_meteors-$release_tag-py3-none-any.whl"
sdist="detect_meteors-$release_tag.tar.gz"

install_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/release-install.XXXXXX")
cd "$install_dir"

for python_version in 3.12 3.13; do
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
