#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [config.yaml]" >&2
  exit 2
fi
config_path="${1:-$project_dir/LLMSimulator/config.yaml}"
if [[ ! -f "$config_path" ]]; then
  echo "Configuration not found: $config_path" >&2
  exit 1
fi
config_path="$(realpath -- "$config_path")"
if [[ ! -x "$project_dir/LLMSimulator/build/run" ]]; then
  echo "Executable not found. Run $project_dir/build.sh first." >&2
  exit 1
fi
mkdir -p "$project_dir/LLMSimulator/data" "$project_dir/LLMSimulator/logs" "$project_dir/LLMSimulator/build/gantt"
cd "$project_dir/LLMSimulator/build"
exec ./run "$config_path"
