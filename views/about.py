import altair as alt
import pandas as pd
import streamlit as st

st.title('About this project')
st.markdown(
    "I wanted to see how well a model can guess a football player's weekly wage from their FIFA 21 stats. "
    'This page is the short version of how it went, including the parts I got wrong.'
)

c1, c2, c3 = st.columns(3)
c1.metric('Players', '18,741')
c2.metric('Clubs', '681')
c3.metric('Typical weekly wage', '€3,000')
st.caption("Data is EA's FIFA 21 player data scraped from sofifa, cleaned in an earlier project. "
           'Wages are weekly, in euros. 237 free agents (no club, wage 0) are left out.')

st.divider()


# ---------- scoreboard ----------
st.header('How it got better')

scores = pd.DataFrame({
    'model': ['Baseline (always guess €3,000)', 'LightGBM', 'LightGBM + log(wage)',
              'LightGBM + club wage', 'LightGBM + club wage + log (final)'],
    'mae': [7754, 3974, 3921, 1533, 1485],
})
scores['final'] = scores['model'].str.contains('final')

bars = alt.Chart(scores).mark_bar(cornerRadiusEnd=4, height=26).encode(
    x=alt.X('mae:Q', title='Average miss on the test set (€ per week)', scale=alt.Scale(domain=[0, 8800])),
    y=alt.Y('model:N', sort=None, title=None, axis=alt.Axis(labelLimit=260)),
    color=alt.condition('datum.final', alt.value('#13663f'), alt.value('#9fb5a8')),
    tooltip=[alt.Tooltip('model:N', title='Model'), alt.Tooltip('mae:Q', title='€ per week', format=',')],
)
labels = bars.mark_text(align='left', dx=6).encode(text=alt.Text('mae:Q', format=','), color=alt.value('#6c7a72'))
st.altair_chart((bars + labels).properties(height=230), width='stretch')

st.markdown(
    '- **Baseline first.** Always guessing the median (€3,000) misses by €7,754 on average. '
    'Any model has to beat that to be worth anything.\n'
    '- **Six algorithms, one wall.** Linear Regression, Random Forest, Gradient Boosting, SVR, XGBoost and LightGBM '
    'all stopped around €4,000. So the algorithm was not the problem.\n'
    '- **The club was the missing piece.** I turned the club into one number, its median wage. '
    'That dropped the miss from €3,974 to €1,533, the biggest jump by far.\n'
    '- **Log on top.** Training on log(wage) makes the model care about percentages instead of euros. '
    'With the club added it won all 5 K-fold rounds, and the average miss in percent went from 137% to 18%.'
)

st.divider()


# ---------- mistakes ----------
st.header('What I got wrong')
st.markdown(
    'After the first models I looked at where the errors were. Cheap players were guessed too high and stars too low, '
    'so I blamed the skewed target and tried log(wage). It barely helped (€3,974 to €3,921).\n\n'
    "The real cause was missing information. Without the club, the model can't tell apart two players "
    'with the same rating at very different clubs, so it guesses in the middle for both. Log was fixing a symptom, '
    'the club fixed the cause.\n\n'
    'Measuring in euros also hid something. In euros the cheapest players looked like the easiest, '
    'but in percent they were the worst, missed by 209% on average.'
)

st.divider()


# ---------- honesty ----------
st.header('Keeping the score honest')
c1, c2 = st.columns(2)
with c1:
    st.subheader('The leak I almost shipped')
    st.markdown(
        "The club wage is built from wages, the thing I'm predicting. If the club medians are calculated "
        'before K-fold splits the data, every round has already seen its own answers.'
    )
    st.metric('Honest K-fold', '€1,596')
    st.metric('Leaky K-fold', '€1,442', '-€154, too good to be true', delta_color='off')
with c2:
    st.subheader('One split can lie')
    st.markdown(
        'Dropping market value looked €16 better on one train/test split. '
        'With 5-fold cross-validation it was worse in 4 of 5 rounds, so I kept it.\n\n'
        'I also used the test set many times to choose between options, which makes it a bit flattering. '
        'The test says €1,485, but the number I trust is the K-fold one.'
    )
    st.metric('Final model, honest miss', '€1,527 per week')

st.divider()


# ---------- limitations ----------
st.header('Limitations')
st.markdown(
    "- **It's game data.** It learns how EA sets wages in FIFA 21, not real contracts.\n"
    '- **Stars are guessed too low.** Messi earns €560,000 and gets about €306,000. In the notebook split, nobody in '
    "the training set earned more than €350,000, and tree models can't predict past what they've seen.\n"
    "- **It needs FIFA stats.** Market value, release clause and every skill rating are inputs, so it can't price "
    'a player who has no FIFA card yet.\n'
    "- **Unknown clubs get the overall median.** The model knows nothing about a club that isn't in the data.\n"
    '- **The what-if is an experiment, not a forecast.** Changing one stat while the rest stay the same can create '
    "players that don't exist, like a 93-rated star at a tiny club."
)

st.divider()


# ---------- next + links ----------
st.header("What I'd try next")
st.markdown(
    '- A league feature. Most of the worst misses are Premier League players, so the league looks like the next missing piece.\n'
    '- A correction for the log pull-down, so top earners stop being guessed too low.\n'
    '- Keeping the test set untouched until the very end, and choosing everything with K-fold only.'
)

st.markdown(
    'The full notebook, from data to final model: '
    '[github.com/GilbertOwen/fifa-player-wage-prediction](https://github.com/GilbertOwen/fifa-player-wage-prediction)'
)
