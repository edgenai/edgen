#!/bin/bash -i
# ubuntu does not source .bashrc in (non-interactive) scripts

# run all tests
# starts the edgen server
# runs tests with pytest
# stops the server

cargo run serve --nogui & > tests/tests.log 2>&1
PID=$!

python --version
python -m pytest tests

kill -2 $PID
