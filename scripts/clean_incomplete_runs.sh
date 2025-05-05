#!/usr/bin/env bash

# Script: clean_invalid_runs.sh
# Purpose: Iterate over each subdirectory under the given root,
#          and prompt to delete if loss_curve.png or train.log is missing.

set -euo pipefail
IFS=$'\n\t'

# Check for exactly one argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/root_dir"
  exit 1
fi

ROOT_DIR="$1"

# Ensure the argument is a directory
if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: '$ROOT_DIR' is not a valid directory."
  exit 1
fi

# Loop over each first-level subdirectory
for dir in "$ROOT_DIR"/*/; do
  # Skip if not a directory
  [ -d "$dir" ] || continue

  # If both required files exist, consider it valid
  if [[ -f "$dir/loss_curve.png" && -f "$dir/train.log" ]]; then
    echo "✔ Valid directory: $(basename "$dir")"
    continue
  fi

  # Otherwise, ask for confirmation before deleting
  read -p "Directory '$(basename "$dir")' is missing loss_curve.png or train.log. Remove it? [y/N] " answer
  case "$answer" in
    [Yy]* )
      echo "Removing: $dir"
      rm -rf "$dir"
      ;;
    * )
      echo "Keeping: $dir"
      ;;
  esac
done

echo "Done."
