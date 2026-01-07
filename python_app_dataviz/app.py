import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Sample Python Dashboard", layout="wide")

st.title("Sample Python Dashboard (Streamlit)")
st.markdown("This dashboard is a template you can run locally or deploy from a GitHub repository.")

# Sidebar: data source
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=True)

@st.cache_data
def load_sample():
    csv = '''date,category,value1,value2,value3
2025-01-01,A,10,100,5
2025-01-02,B,20,90,15
2025-01-03,A,30,80,25
2025-01-04,B,25,70,35
2025-01-05,C,15,60,45
2025-01-06,C,18,50,55
2025-01-07,A,22,40,65
2025-01-08,B,28,30,75
2025-01-09,C,35,20,85
2025-01-10,A,40,10,95
'''
    return pd.read_csv(StringIO(csv), parse_dates=["date"])

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=[0])
elif use_sample:
    df = load_sample()
else:
    st.info("Upload a CSV or enable 'Use sample data' to view the dashboard.")
    st.stop()

st.sidebar.write("Rows:", len(df))
st.dataframe(df.head(50))

# Controls
st.sidebar.header("Controls")
category_filter = st.sidebar.multiselect("Category", options=sorted(df['category'].unique()), default=sorted(df['category'].unique()))
date_range = st.sidebar.date_input("Date range", value=(df['date'].min(), df['date'].max()))

mask = df['category'].isin(category_filter) & (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
dff = df.loc[mask].copy()

# Layout: 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Time series — value1")
    fig1 = px.line(dff, x='date', y='value1', color='category', markers=True, title="Value1 over time")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Histogram — value2")
    fig2 = px.histogram(dff, x='value2', nbins=10, title="Distribution of Value2")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Summary statistics")
    st.write(dff[['value1','value2','value3']].describe())

with col2:
    st.subheader("Bar chart — aggregated value1 by category")
    agg = dff.groupby('category', as_index=False).agg({'value1':'sum','value2':'mean'})
    fig3 = px.bar(agg, x='category', y='value1', title="Sum of value1 by category", text='value1')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Scatter — value1 vs value3")
    fig4 = px.scatter(dff, x='value1', y='value3', color='category', size='value2', title="Value1 vs Value3")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Matplotlib example — boxplot of value1 by category")
    fig, ax = plt.subplots()
    dff.boxplot(column='value1', by='category', ax=ax)
    ax.set_title("Boxplot of value1 by category")
    ax.set_xlabel("")
    st.pyplot(fig)

# Assignment outputs viewer
st.sidebar.header("Assignment outputs")
show_outputs = st.sidebar.checkbox("Show generated visualizations", value=False)
if show_outputs:
    import glob
    import os
    from streamlit.components.v1 import html as st_html

    files = sorted(glob.glob('output/*'))
    st.write("### Generated outputs")

    if not files:
        st.info("No files found in the `output/` directory. Run the generator or click the button below.")
    else:
        choice = st.selectbox("Select a file to preview", files, format_func=lambda x: os.path.basename(x))
        if choice.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
            st.image(choice, use_column_width=True)
        elif choice.lower().endswith('.html'):
            with open(choice, 'r', encoding='utf-8') as f:
                html_str = f.read()
            st_html(html_str, height=700)
        else:
            st.write(f"Preview not supported for: `{choice}`")

    if st.button("Generate visualizations"):
        import subprocess, sys, os
        with st.spinner("Generating visualizations..."):
            script_candidates = ["python_app_dataviz.py", "data_viz_assignment.py"]
            script = next((s for s in script_candidates if os.path.exists(s)), None)
            if script is None:
                st.error(f"No generator script found. Expected one of: {', '.join(script_candidates)}")
            else:
                proc = subprocess.run([sys.executable, script], capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success(f"Generation finished by running {script} — refreshing to show new files.")
                    st.experimental_rerun()
                else:
                    st.error(f"Generation failed when running {script}. See logs below.")
                    st.code(proc.stderr or proc.stdout)

# Assignment gallery (main page)
st.header("Assignment gallery")
if st.checkbox("Show gallery of generated visualizations on main page"):
    import glob, os
    from streamlit.components.v1 import html as st_html

    files = sorted(glob.glob('output/*'))
    if not files:
        st.info("No outputs found in the `output/` directory. Use the sidebar 'Generate visualizations' button to create them.")
    else:
        st.write(f"Found {len(files)} files in `output/`:")
        ncols = 3
        cols = st.columns(ncols)
        for i, f in enumerate(files):
            col = cols[i % ncols]
            name = os.path.basename(f)
            mtime = os.path.getmtime(f)
            with col:
                st.markdown(f"**{name}**")
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                    st.image(f, caption=name, use_column_width=True)
                elif f.lower().endswith('.html'):
                    with st.expander(name):
                        try:
                            with open(f, 'r', encoding='utf-8') as fh:
                                html_str = fh.read()
                            st_html(html_str, height=400)
                        except Exception as e:
                            st.write(f"Unable to preview HTML: {e}")
                else:
                    st.write(name)
        if st.button("Refresh gallery"):
            st.experimental_rerun()

st.markdown("""---
## How to use / deploy
- To run locally:
  1. Install dependencies: `pip install -r requirements.txt`
  2. Run: `streamlit run app.py`
- To deploy from GitHub: Connect this repository to Streamlit Community Cloud (https://streamlit.io/cloud) or configure a platform like Heroku / Render.
""")
