# start the app with: streamlit run app.py
import streamlit as st

st.set_page_config(page_title='FIFA 21 Wage Estimator', page_icon='⚽', layout='wide')

page = st.navigation([
    st.Page('views/main.py', title='Wage Estimator', icon=':material/payments:', default=True),
    st.Page('views/about.py', title='About', icon=':material/info:'),
])
page.run()
