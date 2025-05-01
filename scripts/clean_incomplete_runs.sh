#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 --root_dir PATH"
    echo "Removes directories that don't contain both loss_curve.png and train.log files"
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_dir) ROOT_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1"; usage;;
    esac
done

if [ -z "$ROOT_DIR" ]; then
    echo "Missing required argument: --root_dir"
    usage
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: $ROOT_DIR is not a valid directory"
    exit 1
fi

echo "Scanning for incomplete runs in $ROOT_DIR..."

# Find all directories that don't have both required files
find "$ROOT_DIR" -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        has_loss_curve=false
        has_train_log=false
        
        if [ -f "$dir/loss_curve.png" ]; then
            has_loss_curve=true
        fi
        
        if [ -f "$dir/train.log" ]; then
            has_train_log=true
        fi
        
        if [ "$has_loss_curve" = false ] || [ "$has_train_log" = false ]; then
            echo "Removing incomplete run: $dir"
            rm -rf "$dir"
        fi
    fi
done

echo "Cleanup complete!" 