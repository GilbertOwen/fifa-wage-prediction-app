# things both pages need: the saved files and the predict function
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from prepare import build_features, add_club_wage, align

here = Path(__file__).parent


@st.cache_resource   # load once, keep it in memory for every click and every user
def load_bundle():
    return joblib.load(here / 'artifacts' / 'model.joblib')


@st.cache_data
def load_raw():
    return pd.read_csv(here / 'data' / 'fifa21_cleaned.csv')


@st.cache_data
def load_players():
    # one row per player, with the honest out-of-fold estimate from train.py
    return pd.read_csv(here / 'artifacts' / 'oof_predictions.csv')


def predict_wage(raw_rows):
    """Raw csv rows in, weekly wage in euros out. Same recipe as training."""
    bundle = load_bundle()
    X, _, info = build_features(raw_rows)
    X = align(X, bundle['feature_columns'])
    X = add_club_wage(X, info['Team'], bundle['club_medians'], bundle['overall_median'])
    return np.expm1(bundle['model'].predict(X))


def euro(v):
    return f'€{v:,.0f}'
