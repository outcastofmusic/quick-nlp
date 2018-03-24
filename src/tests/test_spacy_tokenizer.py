import subprocess

import pytest

from quicknlp.data import SpacyTokenizer


@pytest.fixture(scope="session")
def spacy_en():
    try:
        import spacy
        spacy.load("en")
    except IOError as e:
        subprocess.run(["python", "-m", "spacy", "download", "en"])


@pytest.fixture()
def tokenizer(spacy_en):
    return SpacyTokenizer()


def test_spacy_tokenizer(tokenizer):
    sentence = "You guys, you guys! Chef is going away. \n"

    expected_results = ["You", "guys", ",", "you", "guys", "!", "Chef", "is", "going", "away", ".", "\n"]
    results = tokenizer(sentence)
    assert len(results) == 12
    assert results == expected_results


# def test_spacy_tokenizer_reverse(tokenizer):
#     tokenizer.reverse = True
#     sentence = "You guys, you guys! Chef is going away. \n"
#
#     expected_results = ["You", "guys", ",", "you", "guys", "!", "Chef", "is", "going", "away", ".", "\n"][::-1]
#     results = tokenizer(sentence)
#     assert len(results) == 12
#     assert results == expected_results


def test_spacy_tokenizer_sentences(tokenizer):
    sentence = "You guys, you guys! Chef is going away. \nGoing away? For how long?\n"

    expected_results = [
        ["You", "guys", ",", "you", "guys", "!"],
        ["Chef", "is", "going", "away", ".", "\n"],
        ["Going", "away", "?"],
        ["For", "how", "long", "?", "\n"]
    ]
    results = tokenizer(sentence, sentence=True)
    assert len(results) == 4
    for index in range(len(results)):
        assert results[index] == expected_results[index]
