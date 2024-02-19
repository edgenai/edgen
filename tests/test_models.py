
from edgen import Edgen, APIConnectionError
from edgen.resources.misc import Version
import pytest
import subprocess

client = Edgen()

def test_models():
    try:
        models = client.models.list()
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    assert(type(models) is list)

if __name__ == "__main__":
   test_models()
