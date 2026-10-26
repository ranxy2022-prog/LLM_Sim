#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake_bin="$project_dir/.venv/bin/cmake"
if [[ ! -x "$cmake_bin" ]]; then
  echo "Missing build tools. Create .venv and install requirements-tools.txt (see SETUP.txt)." >&2
  exit 1
fi
"$cmake_bin" -S "$project_dir/LLMSimulator" -B "$project_dir/LLMSimulator/build" \
  -G Ninja -DCMAKE_MAKE_PROGRAM="$project_dir/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20
"$cmake_bin" --build "$project_dir/LLMSimulator/build" --parallel "${BUILD_JOBS:-8}"
