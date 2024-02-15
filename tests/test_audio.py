
from pathlib import Path
import Levenshtein

from edgen import Edgen, APIConnectionError
import pytest

expected = ' The woods are lovely, dark and deep, ' \
           'but I have promises to keep ' \
           'and miles to go before I sleep, ' \
           'and miles to go before I sleep.'

client = Edgen()

def test_transcriptions():

    speech_file_path = Path(__file__).parent.parent / "crates" / "edgen_server" / "resources" / "frost.wav"

    try:
        transcription = client.audio.transcriptions.create(
            model = "default",
            file = speech_file_path,
        )
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    print(transcription)
    print(expected)

    have = transcription.text

    assert(type(have) is str)

    d = Levenshtein.distance(have, expected)
    similarity = 100 - ((d / len(expected)) * 100)
    print(f"distance: {d} of '{have}', similarity: {similarity}")

    assert(similarity > 90.0)

if __name__ == "__main__":
   test_transcriptions()
