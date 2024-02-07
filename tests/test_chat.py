
from edgen import Edgen, APIConnectionError
import pytest

client = Edgen()

def test_completions_streaming():
    try:
        stream = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "What is the result of 1+2?",
                },
            ],
            stream=True,
        )
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    assert(stream.response.status_code == 200)

    answer = ""
    for chunk in stream:
        if not chunk.choices:
            continue

        answer += chunk.choices[0].delta.content

    # print(answer)
    assert(type(answer) is str)

"""
def test_completions():
    try:
        answer = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "What is the result of 1+2?",
                },
            ],
        )
    except APIConnectionError:
        pytest.fail("No connection. Is edgen running?")

    assert(answer.response.status_code == 200)

    print(answer)
    assert(type(answer) is str)
"""

if __name__ == "__main__":
   test_completions_streaming()
