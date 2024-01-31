#!/bin/bash -i
# ubuntu does not source .bashrc in (non-interactive) scripts

# run all tests:
# starts the edgen server
# runs tests with pytest
# stops the server

echo "================================================"
date
cargo run version
echo "================================================"

cargo run serve --nogui & > tests/tests.log 2>&1
PID=$!

sleep 1

python -m pytest tests

kill -2 $PID
