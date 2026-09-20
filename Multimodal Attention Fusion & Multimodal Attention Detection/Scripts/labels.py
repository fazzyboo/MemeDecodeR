"""
Canonical label handling, shared by the MAF and MuLAD pipelines.

The integer codes follow the MAF paper's ordering so that MMAE - which treats the labels
as an ordinal scale - stays comparable with the published numbers:

    NoAg = 0, GAg = 1, PAg = 2, RAg = 3, Oth = 4

The number of classes is resolved from the data rather than hardcoded. The full MIMOSA
corpus carries all five; a subset that lacks "others" is a valid four-way task and is
handled without edits. Anything that leaves a hole in the code sequence (e.g. classes
{0,1,2,4}) is rejected, because the model's output layer assumes contiguous indices.
"""
CANONICAL = [
    ("non-aggressive", "NoAg"),
    ("gendered aggression", "GAg"),
    ("political aggression", "PAg"),
    ("religious aggression", "RAg"),
    ("others", "Oth"),
]

LABEL_MAP = {name: idx for idx, (name, _) in enumerate(CANONICAL)}
ALL_TARGET_NAMES = [short for _, short in CANONICAL]

# Alternative spellings seen in the competition-era CSVs, mapped onto the canonical names.
ALIASES = {
    "neutral": "non-aggressive",
    "nonaggressive": "non-aggressive",
    "non aggressive": "non-aggressive",
    "genders": "gendered aggression",
    "gendered": "gendered aggression",
    "politics": "political aggression",
    "political": "political aggression",
    "religion": "religious aggression",
    "religious": "religious aggression",
    "other": "others",
}


def normalise(label):
    """Map a raw CSV label string onto a canonical class name."""
    key = str(label).strip().lower()
    return ALIASES.get(key, key)


def encode(frame, name):
    """Return a copy of `frame` with Captions NaN-filled and Label as canonical integers."""
    frame = frame.copy()
    frame["Captions"] = frame["Captions"].fillna("").astype(str)
    frame["Label"] = frame["Label"].map(lambda v: LABEL_MAP.get(normalise(v)))
    if frame["Label"].isna().any():
        bad = frame.loc[frame["Label"].isna(), "Label"].head()
        raise ValueError("Unmapped label(s) in {}:\n{}".format(name, bad))
    frame["Label"] = frame["Label"].astype(int)
    return frame.reset_index(drop=True)


def resolve_classes(frames):
    """Infer (num_classes, target_names) from the encoded splits actually loaded."""
    present = sorted({int(v) for frame in frames for v in frame["Label"].unique()})
    if present != list(range(len(present))):
        raise ValueError(
            "Label codes {} are not contiguous from 0; the output layer cannot be sized. "
            "Check the Label column against labels.CANONICAL.".format(present)
        )
    return len(present), [ALL_TARGET_NAMES[i] for i in present]
