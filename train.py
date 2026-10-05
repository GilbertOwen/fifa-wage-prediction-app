# run once: python train.py
# makes artifacts/model.joblib and artifacts/oof_predictions.csv for the app
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from prepare import build_features, fit_club_medians, add_club_wage

here = Path(__file__).parent
raw = pd.read_csv(here / 'data' / 'fifa21_cleaned.csv')
X, y, info = build_features(raw)
teams = info['Team']
print('players:', len(X), '| features:', X.shape[1])


# out-of-fold predictions, same idea as the honest loop in the notebook.
# every player gets predicted by a model that never saw their wage,
# so the player lookup in the app shows real errors, not memorised ones
oof = pd.Series(np.nan, index=X.index)
folds = KFold(n_splits=5, shuffle=True, random_state=42)

for fit_pos, check_pos in folds.split(X):
    fit_idx, check_idx = X.index[fit_pos], X.index[check_pos]

    # club medians rebuilt every round, from that round's training rows only
    medians, overall = fit_club_medians(teams.loc[fit_idx], y.loc[fit_idx])
    X_fit = add_club_wage(X.loc[fit_idx], teams.loc[fit_idx], medians, overall)
    X_check = add_club_wage(X.loc[check_idx], teams.loc[check_idx], medians, overall)

    model = LGBMRegressor(verbose=-1).fit(X_fit, np.log1p(y.loc[fit_idx]))
    oof.loc[check_idx] = np.expm1(model.predict(X_check))

print('out-of-fold MAE:', round(mean_absolute_error(y, oof)))


# final model for the what-if, trained on everyone
medians, overall = fit_club_medians(teams, y)
X_all = add_club_wage(X, teams, medians, overall)
model = LGBMRegressor(verbose=-1).fit(X_all, np.log1p(y))

(here / 'artifacts').mkdir(exist_ok=True)
joblib.dump({
    'model': model,
    'feature_columns': list(X.columns),   # the 88, before club_wage gets added
    'club_medians': medians,
    'overall_median': overall,
}, here / 'artifacts' / 'model.joblib')

info.assign(predicted=oof).to_csv(here / 'artifacts' / 'oof_predictions.csv', index=False)
print('saved to artifacts/')
