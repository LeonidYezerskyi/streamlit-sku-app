import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

# Налаштування сторінки
st.set_page_config(
    page_title="SKU Portfolio Dashboard",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("📊 SKU Portfolio Dashboard")
st.markdown("---")

# Завантаження даних
@st.cache_data
def load_data():
    # Тут буде завантаження вашого JSON файлу
    return None

# Основна навігація
tab1, tab2 = st.tabs(["📈 Portfolio Overview", "🔍 SKU Details"])

with tab1:
    st.header("Portfolio Overview")
    st.write("Тут буде портфоліо огляд")

with tab2:
    st.header("SKU Details")
    st.write("Тут буде детальна інформація по SKU")
