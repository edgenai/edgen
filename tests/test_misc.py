
from edgen import Edgen, APIConnectionError
from edgen.resources.misc import Version
import pytest
import subprocess

client = Edgen()

def test_version():
    expected = edgen_version()

    try:
        version = client.misc.version.create()
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    # print(version)
    # print(f"{expected} == {format_version(version)}")   

    assert(type(version) is Version)
    assert(format_version(version) == expected)

def edgen_version():
    finished = subprocess.run(["cargo", "run", "version"], capture_output=True, text=True)
    version = ''.join(f"{finished.stdout}".splitlines())
    return version

def format_version(version):
    build = ''
    if version.build: 
        build = '-'.join(version.build)
    return f"{version.major}.{version.minor}.{version.patch}{build}"

if __name__ == "__main__":
   test_version()
