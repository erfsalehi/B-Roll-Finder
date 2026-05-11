import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'Title': ['https://google.com/#title=Google_Search', 'https://yahoo.com/#title=Yahoo_Mail']
})

st.data_editor(
    df,
    column_config={
        "Title": st.column_config.LinkColumn("Title", display_text=r"#title=(.*)")
    }
)
