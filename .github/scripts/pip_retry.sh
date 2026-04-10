#!/usr/bin/env bash

retry_pip() {
  local attempt=1
  until python -m pip "$@"; do
    if [ "$attempt" -ge 3 ]; then
      echo "pip $* failed after ${attempt} attempts"
      return 1
    fi
    echo "pip $* failed on attempt ${attempt}; retrying..."
    attempt=$((attempt + 1))
  done
}
