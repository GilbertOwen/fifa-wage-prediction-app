import pandas as pd
import streamlit as st

from shared import load_players, load_raw, load_bundle, predict_wage, euro

players = load_players()
raw = load_raw()
clubs = sorted(load_bundle()['club_medians'].index)

st.title('FIFA 21 Wage Estimator')
st.caption("Weekly wages in euros, from EA's FIFA 21 player data. "
           "The model is LightGBM, trained on log(wage) with each club's median wage as a feature.")


# ---------- player lookup ----------
labels = players['display_name'] + ' · ' + players['Team'] + ' · ' + players['BP'] + ' ' + players['OVA'].astype(str)
pick = st.selectbox('Player', players.index, format_func=lambda i: labels[i], key='player')
p = players.loc[pick]

st.markdown(f"**{p['display_name']}** · {p['Team']} · {p['BP']} · OVA {p['OVA']} · "
            f"age {p['Age']} · market value {euro(p['Value'])}")

off = (p['predicted'] - p['Wage']) / p['Wage']
c1, c2 = st.columns(2)
c1.metric('Actual weekly wage', euro(p['Wage']))
c2.metric('Model estimate', euro(p['predicted']), f'{off:+.0%} vs actual', delta_color='off')
st.caption("This estimate comes from a model that never saw this player's wage (5-fold out-of-fold), "
           'so the miss is real. Across all players the typical miss is about €1,440 a week.')

st.divider()


# ---------- what-if ----------
st.subheader('What if?')
raw_row = raw[raw['ID'] == p['ID']]

# keys include the player ID, so the controls reset when you pick another player
c1, c2, c3 = st.columns(3)
club = c1.selectbox('Club', clubs, index=clubs.index(p['Team']), key=f"wi_club_{p['ID']}")
ova = c2.slider('OVA', 47, 93, int(p['OVA']), key=f"wi_ova_{p['ID']}")
age = c3.slider('Age', 16, 53, int(p['Age']), key=f"wi_age_{p['ID']}")

changed = raw_row.assign(Team=club, OVA=ova, Age=age)
before, after = predict_wage(pd.concat([raw_row, changed], ignore_index=True))

diff = after - before
c1, c2 = st.columns(2)
c1.metric('As he is', euro(before))
c2.metric('With your changes', euro(after), f"{'+' if diff >= 0 else '-'}€{abs(diff):,.0f}")
st.caption("Uses the final model trained on every player, so 'as he is' can differ from the estimate above. "
           'Every other stat (market value, skills) stays the same, so this shows how the model reacts, '
           "not a realistic player. Trees can't go past what they've seen, so a star at a tiny club gives silly numbers.")

st.divider()


# ---------- above / below the estimate ----------
st.subheader('Paid above or below the estimate')
options = ['All clubs'] + clubs
which = st.selectbox('Club', options, index=options.index(p['Team']), key=f"gap_club_{p['ID']}")

view = players if which == 'All clubs' else players[players['Team'] == which]
view = view.assign(gap=view['Wage'] - view['predicted'])

cols = {'display_name': 'Player', 'BP': 'Pos', 'OVA': 'OVA', 'Age': 'Age',
        'Wage': 'Actual €/wk', 'predicted': 'Estimate €/wk', 'gap': 'Gap €/wk'}
money = st.column_config.NumberColumn(format='localized')


def show(df):
    st.dataframe(df[list(cols)].rename(columns=cols).round(0), hide_index=True, width='stretch',
                 column_config={'Actual €/wk': money, 'Estimate €/wk': money, 'Gap €/wk': money})


above, below = st.tabs(['Paid above the estimate', 'Paid below the estimate'])
with above:
    show(view.nlargest(10, 'gap'))
with below:
    show(view.nsmallest(10, 'gap'))
st.caption("'Above' doesn't mean overpaid in real life. It means the wage isn't explained by what the model sees, "
           'like fame, contract history or the league.')
