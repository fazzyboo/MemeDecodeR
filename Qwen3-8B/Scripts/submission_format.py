"""
Competition submission format, shared by generate_submission.py and
predictions_to_submission.py so the two can never disagree.

sample_submission.csv expects the ORIGINAL Train/Train.csv vocabulary (Neutral, Genders,
Politics, Religion), not the paper's labels used internally. The mapping is keyed by the
short codes in dataset.TARGET_NAMES, so it never depends on remembering an index order.
"""

SHORT_TO_SUBMISSION = {"NoAg": "Neutral", "GAg": "Genders", "PAg": "Politics", "RAg": "Religion"}
SUBMISSION_VOCAB = set(SHORT_TO_SUBMISSION.values())


def validate_submission(out_df, sample_df):
    """Raise if out_df would not be accepted as a drop-in replacement for sample_df."""
    id_col, target_col = sample_df.columns[0], sample_df.columns[1]
    if list(out_df.columns) != list(sample_df.columns):
        raise ValueError(
            "columns {} do not match sample_submission.csv {}".format(
                list(out_df.columns), list(sample_df.columns)
            )
        )
    if list(out_df[id_col]) != list(sample_df[id_col]):
        raise ValueError("image names or row order differ from sample_submission.csv")
    if out_df[target_col].isna().any():
        raise ValueError("{} rows have no prediction".format(int(out_df[target_col].isna().sum())))
    unexpected = set(out_df[target_col]) - SUBMISSION_VOCAB
    if unexpected:
        raise ValueError("labels outside the competition vocabulary: {}".format(sorted(unexpected)))
