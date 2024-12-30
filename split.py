#!/usr/bin/env python3
import os
import pandas as pd

TEST_FRAC = 0.1
TRAIN_FRAC = 0.9

def save_split (root, prefix, df):
    df.to_csv(os.path.join(root, f'{prefix}_labels.csv'), index=False)

def save (path, df_train, df_test):
    os.makedirs(path, exist_ok=True)
    save_split(path, 'train', df_train)
    save_split(path, 'val', df_test)

def main ():
    df = pd.read_csv('data/train_labels.csv')
    l0 = df['diagnosis_control']
    l1 = df['diagnosis_mci']
    l2 = df['diagnosis_adrd']
    assert ((l0 == 0) | (l0 == 1)).all()
    assert ((l1 == 0) | (l1 == 1)).all()
    assert ((l2 == 0) | (l2 == 1)).all()
    assert (l0 + l1 + l2 == 1).all()
    df['label'] = (0 * l0 + 1 * l1 + 2 * l2).astype(int)
    df = df[['uid', 'label']]
    save_split('data', 'all', df)
    # sample 10% for testing
    test_df = df.sample(frac=TEST_FRAC, random_state=2024)
    save_split('data', 'validation', test_df)
    #df = df.drop(test_df.index)
    # Perform random split into training and test sets
    for i, seed in enumerate(range(42, 42+40)):
        print("="*20, f"Split {i}")
        train_df = df.sample(frac=TRAIN_FRAC, random_state=seed)
        val_df = df.drop(train_df.index)
        save(f'data/split{i}', train_df, val_df)
        print(f"Training set shape: {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")

if __name__ == '__main__':
    main()

