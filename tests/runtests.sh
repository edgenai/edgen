#!/bin/bash -i
# ubuntu does not source .bashrc in (non-interactive) scripts

# run all tests:
# starts the edgen server
# runs tests with pytest
# stops the server

cargo build --release

echo "================================================"
date
target/release/edgen version
echo "================================================"

target/release/edgen serve --nogui & > tests/tests.log 2>&1
PID=$!

sleep 1

python -m pytest tests

kill -2 $PID
