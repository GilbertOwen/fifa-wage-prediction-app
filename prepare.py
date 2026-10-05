import pandas as pd

# everything we decided against in the notebook (section 3)
drop_cols = [
    # identifiers, they tell you who not why they're paid
    'photoUrl', 'playerUrl', 'LongName', 'Name', 'ID',

    # checked, no relationship with Wage
    'foot', 'Height', 'Weight',

    # too many categories (Team comes back later as club_wage)
    'Nationality', 'Team',

    # covered by other columns
    'Positions', 'Joined', 'Contract_type', 'Loan Date End',
    'Contract_start', 'Contract_end',

    # sums of columns we're keeping
    'Attacking', 'Skill', 'Movement', 'Power', 'Mentality',
    'Defending', 'Goalkeeping', 'Total Stats', 'Base Stats',
]


def build_features(raw):
    """Same steps as the notebook, sections 3b to 4. Returns X, y and info (for showing players)."""
    df = raw.copy()

    df['contract_length'] = df['Contract_end'] - df['Contract_start']
    df['years_at_club'] = 2021 - df['Contract_start']

    # free agents earn 0 and have no club, the model isn't meant for them
    df = df[df['Contract_type'] != 'Free'].copy()

    df['on_loan'] = (df['Contract_type'] == 'On Loan').astype(int)
    df['contract_length'] = df['contract_length'].fillna(0)
    df['years_at_club'] = df['years_at_club'].fillna(0)

    info = df[['ID', 'Name', 'LongName', 'Team', 'BP', 'Age', 'OVA', 'Value', 'Wage']].copy()
    # 7 names are already broken in EA's raw file, LongName is fine for those
    broken = info['Name'].str.contains('�')
    info['display_name'] = info['Name'].mask(broken, info['LongName'])

    df = df.drop(columns=['BOV', 'Growth'])
    df = df.drop(columns=drop_cols)
    df = pd.get_dummies(df, columns=['BP', 'A/W', 'D/W'], dtype=int)

    X = df.drop(columns='Wage')
    y = df['Wage']
    return X, y, info


def fit_club_medians(teams, wages):
    """Median weekly wage per club, learned only from the rows we pass in."""
    return wages.groupby(teams).median(), wages.median()


def add_club_wage(X, teams, medians, overall):
    # a club we've never seen gets the overall median instead of NaN
    return X.assign(club_wage=teams.map(medians).fillna(overall))


def align(X, feature_columns):
    # one player only gets its own one-hot columns (BP_ST, say),
    # this adds the missing ones as 0 and puts everything in the training order
    return X.reindex(columns=feature_columns, fill_value=0)
