import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def safe_numeric(s):
    try:
        return pd.to_numeric(s)
    except Exception:
        return pd.to_numeric(pd.Series(s), errors='coerce')


def parse_wl(s):
    # parse strings like '22-2' -> wins, losses
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.search(r"(\d+)[^\d]+(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # sometimes single number
    m2 = re.search(r"(\d+)", str(s))
    if m2:
        return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)


def load_and_process(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Target column
    target_col = 'Overall Seed'

    # Keep only rows with numeric target
    df_train[target_col] = pd.to_numeric(df_train[target_col], errors='coerce')
    df_train = df_train[df_train[target_col].notna()].copy()

    # Candidate numeric columns
    numeric_cols = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
                    'NETSOS', 'NETNonConfSOS', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']

    for col in numeric_cols:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # Parse WL into wins and losses
    for df in (df_train, df_test):
        wins = []
        losses = []
        for v in df.get('WL', pd.Series()).fillna(''):
            w, l = parse_wl(v)
            wins.append(w)
            losses.append(l)
        df['WL_wins'] = wins
        df['WL_losses'] = losses

    # Conference and Bid Type encoding (label)
    for col in ['Conference', 'Bid Type']:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('NA')
            df_test[col] = df_test[col].fillna('NA')
            # simple label encoding
            cats = pd.concat([df_train[col], df_test[col]]).astype('category')
            mapping = {c: i for i, c in enumerate(cats.cat.categories)}
            df_train[col + '_enc'] = df_train[col].map(mapping).fillna(-1)
            df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1)

    # Build feature list
    features = []
    for col in numeric_cols:
        if col in df_train.columns:
            features.append(col)
    features += ['WL_wins', 'WL_losses', 'Conference_enc', 'Bid Type_enc']

    # Fill NaNs with median
    for col in features:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(med)

    return df_train, df_test, features, target_col


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print('Loading and processing data...')
    df_train, df_test, features, target_col = load_and_process(train_path, test_path)

    X = df_train[features].values
    y = df_train[target_col].values
    groups = df_train['Season'].values if 'Season' in df_train.columns else None

    print(f'Training rows: {X.shape[0]}, features: {len(features)}')

    # Cross-validation by season
    gkf = GroupKFold(n_splits=5)
    rmses = []
    models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f'Fold {fold+1}')
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            n_jobs=4
        )

        # Fit without early stopping to be compatible with older xgboost
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f'  RMSE: {rmse:.4f}')
        rmses.append(rmse)
        models.append(model)

    print('CV RMSE mean:', np.mean(rmses), 'std:', np.std(rmses))

    # Ensemble prediction on test: average models
    print('Predicting test set...')
    X_test = df_test[features].values
    preds = np.column_stack([m.predict(X_test) for m in models]).mean(axis=1)

    # Map to submission template
    sub = pd.read_csv(submission_template)
    sub = sub.merge(df_test[['RecordID']], on='RecordID', how='right') if 'RecordID' in df_test.columns else sub
    # Align order of template
    sub = sub[['RecordID']].copy()
    sub['Overall Seed'] = preds[:len(sub)]

    sub.to_csv('my_submission.csv', index=False)
    print("Wrote my_submission.csv")


if __name__ == '__main__':
    main()
