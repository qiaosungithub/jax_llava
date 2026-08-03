"""VQA-v2 answer normalization and leave-one-out accuracy.

The normalization table and punctuation behavior intentionally follow the
official GT-Vision-Lab evaluator, including its unanimous-ground-truth branch.
"""

import re


# pylint: disable=line-too-long
REPLACEMENTS = {
    # Contractions from the official VQA evaluator.
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
    # Number words from the official VQA evaluator.
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
# pylint: enable=line-too-long

PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]
ARTICLES = {"a", "an", "the"}


def stripspace_vqav2(text):
    return text.replace("\n", " ").replace("\t", " ").strip()


def postprocess_vqav2_text(text):
    """Normalize one answer exactly like the official VQA-v2 evaluator."""
    has_digit_comma = re.search(r"(\d)(\,)(\d)", text) is not None
    out = text
    for punctuation in PUNCT:
        if has_digit_comma or f"{punctuation} " in text or f" {punctuation}" in text:
            out = out.replace(punctuation, "")
        else:
            out = out.replace(punctuation, " ")
    # Keep the official regex verbatim. In particular, it removes the period
    # in "5." while preserving the decimal point in "5.0".
    out = re.sub(r"(?!<=\d)(\.)(?!\d)", "", out, flags=re.UNICODE)
    words = []
    for word in out.lower().split():
        if word not in ARTICLES:
            words.append(REPLACEMENTS.get(word, word))
    return " ".join(words)


def vqa_accuracy_one(answer, gt_answers, *, normalize_unanimous=False):
    """Return official ten-annotator leave-one-out VQA accuracy.

    The official VQA-v2 evaluator normalizes answers only when the stripped GT
    strings are not unanimous. TextVQA uses the same text normalization but
    applies it unconditionally, exposed through ``normalize_unanimous=True``.
    """
    if not gt_answers or len(gt_answers) < 10:
        return 0.0
    gt_answers = [stripspace_vqav2(value) for value in gt_answers[:10]]
    answer = stripspace_vqav2(answer)
    if normalize_unanimous or len(set(gt_answers)) > 1:
        answer = postprocess_vqav2_text(answer)
        gt_answers = [postprocess_vqav2_text(value) for value in gt_answers]

    matches = [answer == value for value in gt_answers]
    accuracies = []
    for leave_out in range(10):
        match_count = sum(matches[:leave_out]) + sum(matches[leave_out + 1 :])
        accuracies.append(min(1.0, match_count / 3.0))
    return sum(accuracies) / len(accuracies)
