"""
Text augmentation for training data improvement.
"""
import random
from typing import List

random.seed(42)

FILLER_WORDS = ["please", "kindly", "urgently", "immediately", "asap"]
OPENERS = ["Hi,", "Hello,", "Hey,", "Dear team,", ""]
CLOSERS = ["", "Thanks.", "Please help.", "Urgent!", "Need help ASAP."]


def add_opener(text: str) -> str:
    opener = random.choice(OPENERS)
    return f"{opener} {text}".strip() if opener else text


def add_closer(text: str) -> str:
    closer = random.choice(CLOSERS)
    return f"{text} {closer}".strip() if closer else text


def random_case(text: str, probability: float = 0.1) -> str:
    words = text.split()
    result = []
    for word in words:
        if random.random() < probability:
            result.append(word.upper())
        else:
            result.append(word)
    return " ".join(result)


def dropout_words(text: str, dropout_rate: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 3:
        return text
    kept = [w for w in words if random.random() > dropout_rate]
    return " ".join(kept) if kept else text


def augment_text(text: str, techniques: int = 2) -> str:
    augmentations = [add_opener, add_closer, random_case, dropout_words]
    selected = random.sample(augmentations, min(techniques, len(augmentations)))
    for fn in selected:
        text = fn(text)
    return text


def augment_dataset(texts: List[str], multiplier: int = 2) -> List[str]:
    augmented = []
    for text in texts:
        for _ in range(multiplier):
            augmented.append(augment_text(text))
    return augmented
