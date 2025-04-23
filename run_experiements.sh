#!/bin/bash



main() {
    echo "Running Stress-Experiements..."
    uv run main.py
}


main "${@}"

