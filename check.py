# checks that prepare.py does exactly what the notebook did: python check.py
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from prepare import build_features, fit_club_medians, add_club_wage, align

here = Path(__file__).parent
raw = pd.read_csv(here / 'data' / 'fifa21_cleaned.csv')
X, y, info = build_features(raw)
print('shape:', X.shape, '(notebook: 18741 players, 88 features)')


# 1. same split as the notebook, the final model should land on the same test MAE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
team_train, team_test = info.loc[X_train.index, 'Team'], info.loc[X_test.index, 'Team']

medians, overall = fit_club_medians(team_train, y_train)
X_train = add_club_wage(X_train, team_train, medians, overall)
X_test = add_club_wage(X_test, team_test, medians, overall)

model = LGBMRegressor(verbose=-1).fit(X_train, np.log1p(y_train))
mae = mean_absolute_error(y_test, np.expm1(model.predict(X_test)))
print('test MAE:', round(mae, 2), '(notebook: 1484.76)')


# 2. one player prepared on their own should match their row from the full batch.
# one player per position, so every BP_ column gets tested
rows = info.groupby('BP').head(1).index
matches = 0
for row in rows:
    one, _, _ = build_features(raw.loc[[row]])
    one = align(one, X.columns)
    matches += one.equals(X.loc[[row]])
print(f'single player == batch: {matches}/{len(rows)}')
