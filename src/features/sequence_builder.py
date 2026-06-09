from sklearn.model_selection import GroupShuffleSplit
import numpy as np


def create_sequences(
    df,
    seq_len,
    feature_cols,
):
    sequences = []
    targets = []

    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id]

        X_engine = engine_data[feature_cols].values

        y_engine = engine_data["rul"].values

        for i in range(len(X_engine) - seq_len):
            sequences.append(X_engine[i : i + seq_len])

            targets.append(y_engine[i + seq_len])

    return (
        np.array(sequences),
        np.array(targets),
    )


def create_group_split_sequences(
    train_df,
    feature_cols,
    seq_len,
    test_size=0.2,
):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42,
    )

    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["engine_id"]))

    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    X_train, y_train = create_sequences(
        train_split,
        seq_len,
        feature_cols,
    )

    X_val, y_val = create_sequences(
        val_split,
        seq_len,
        feature_cols,
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val,
    )
