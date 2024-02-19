
from edgen import Edgen, APIConnectionError
from edgen.pagination import SyncPage
from edgen.resources.models import Model
import pytest
import subprocess

client = Edgen()

def test_models():
    try:
        models = client.models.list()
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    assert(type(models) is SyncPage[Model])

    for model in models:
        print(model)

if __name__ == "__main__":
   test_models()
