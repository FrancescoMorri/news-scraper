from datetime import datetime
import json
import os
from main_scraper import get_news_and_analyze
from plotters import plot_top_grams, quick_snapshot_charts
import streamlit as st
import pandas as pd
import requests
import base64
import altair as alt


# --------- GitHub API stuff ---------

GITHUB_TOKEN = st.secrets["github_token"]
REPO = "francescomorri/news-scraper"
FILE_PATH = "daily_data.jsonl"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"


def load_today_or_false(date):
    stored_data = None
    # get actual file from repository
    res = requests.get(API_URL, headers={"Authorization": f"token {GITHUB_TOKEN}"})
    data = res.json()
    content = base64.b64decode(data['content']).decode('utf-8')
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines
        if obj.get("date") == date:
            stored_data = obj
            break
    # if os.path.exists(path):
    #     with open(path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             try:
    #                 obj = json.loads(line)
    #             except json.JSONDecodeError:
    #                 continue  # skip malformed lines
    #             if obj.get("date") == date:
    #                 stored_data = obj
    #                 break
    # else:
    #     return False
    return stored_data, content, data


DEFAULT_PATH = "daily_data.jsonl"

st.set_page_config(page_title="Daily News Dashboard", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.header("Settings")
path = st.sidebar.text_input("Data file (JSONL)", value=DEFAULT_PATH)
force_rescrape = st.sidebar.checkbox("Force re-scrape today", value=False)
show_raw_record = st.sidebar.checkbox("Show raw JSON record", value=False)
today = datetime.today().strftime('%Y-%m-%d')

# -------- TABS ---------
daily_tab, history_tab = st.tabs(["Daily Dashboard", "History"])


with daily_tab:
    # ---------- LOAD OR SCRAPE ----------
    st.title("Daily News Dashboard")
    st.subheader(f"Scraping and analyzing news for {today}, news headlines coming from: [Timef](https://timef.com/business/), [skimfeed](https://skimfeed.com/custom.php?f=l%2Cp%2C119%2C121%2C122%2C123%2C124%2C125%2C126%2C127%2C156) and [brutalist.report](https://brutalist.report/topic/business). The analysis is done using a combination of _VADER sentiment analysis_ and the _Loughran-McDonald_ financial sentiment lexicon.")
    status_placeholder = st.empty()


    stored_data = None
    used_cached = False

    if not force_rescrape:
        status_placeholder.info("Checking for today's record…")
        stored_data, content, data = load_today_or_false(date=today)  # returns the current data or None, and the full file content
        if stored_data is not None:
            used_cached = True
            status_placeholder.success("Loaded today's record from JSONL.")
        else:
            status_placeholder.warning("No record for today found. Scraping now…")
            stored_data = None

    if stored_data is None:
        # Either forced re-scrape or not found
        stored_data = get_news_and_analyze()
        status_placeholder.info("Scraping and analysis done. Saving data…")
        # if not forced_rescrape append today's data to daily_data.jsonl
        if not force_rescrape:
            # if os.path.exists(path):
            #     mode = "a"
            # else:
            #     mode = "w"
            # with open(path, mode, encoding="utf-8") as f:
            #     f.write(json.dumps(stored_data) + "\n")
            updated_file = content + json.dumps(stored_data) + "\n"
            res = requests.put(API_URL, json={
                "message": "Update JSONL via Streamlit",
                "content": base64.b64encode(updated_file.encode()).decode(),
                "sha": data["sha"]
            }, headers={"Authorization": f"token {GITHUB_TOKEN}"})
            status_placeholder.success("Scraping and analysis complete. Data saved to JSONL.")
        else:
            # remove any existing record for today and append the new one
            outpath = path + "-tmp"
            with open(path, "r") as infile, open(outpath, "w") as outfile:
                for line in infile:
                    record = json.loads(line)
                    if record["date"] == today:
                        record = stored_data  # overwrite this one
                    outfile.write(json.dumps(record) + "\n")
            if os.path.exists(path):
                os.remove(path)
            os.rename(outpath, path)
            status_placeholder.success("Re-scrape and analysis complete. Data updated in JSONL.")



    # here we have data in a json format no matter what happened, so we can plot and print everything
    # Unigrams and bigrams
    st.subheader("Unigrams and Bigrams")
    col1, col2 = st.columns(2)
    with col1:
        fig_uni = plot_top_grams(pd.DataFrame(stored_data['top_unigrams']), name="unigrams")
        st.pyplot(fig_uni)
    with col2:
        fig_bi = plot_top_grams(pd.DataFrame(stored_data['top_bigrams']), name="bigrams")
        st.pyplot(fig_bi)


    # ---------- HEADLINES ----------
    st.subheader("Top 5 News Headlines")
    st.markdown(f"### :red[Most negative]: {stored_data['most_negative']}")
    st.markdown(f"### :green[Most positive]: {stored_data['most_positive']}")
    st.markdown(f"### :blue[Most uncertainty]: {stored_data['most_uncertain']}")
    st.markdown(f"### :orange[Most litigious]: {stored_data['most_litigious']}")
    st.markdown(f"### :violet[Most constraining]: {stored_data['most_constraining']}")


    # ---------- Vader and LM ----------
    st.subheader("LM and Vader Snapshot Charts")
    fig_m1, fig_m2 = quick_snapshot_charts(stored_data)
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(fig_m1)
    with col4:
        st.pyplot(fig_m2)


    # ---------- Top LM terms ----------
    st.subheader("Top LM Terms")
    top_negative_df = stored_data['top_lm_negative']
    top_positive_df = stored_data['top_lm_positive']
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Top LM-Negative Terms**")
        st.table(top_negative_df)
    with col6:
        st.markdown("**Top LM-Positive Terms**")
        st.table(top_positive_df)

    #print("Top LM-negative terms:")
    #print(tops["negative"].head(10))
    #print("\nTop LM-uncertainty terms:")
    #print(tops["uncertainty"].head(10))
    #print("\nTop LM-positive terms:")
    #print(tops["positive"].head(10))

    # ---------- RAW RECORD ----------
    if show_raw_record:
        st.subheader("Raw JSON Record")
        st.json(stored_data)


with history_tab:
    st.title("Historical Data")
    st.subheader("Analysis of past records stored in the JSONL file.")

    # load all data from JSONL
    _, content, _ = load_today_or_false(date="")  # we just want the full content
    
    hist_pct_neg = []
    hist_pct_unc = []
    hist_pct_pos = []
    hist_bigrams = []
    dates = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines
        dates.append(obj['date'])
        hist_pct_neg.append(obj['head_w_neg'])
        hist_pct_unc.append(obj['head_w_unc'])
        hist_pct_pos.append(obj['head_w_pos'])
        hist_bigrams.append(obj['top_bigrams'][0]['term'])
    hist_df = pd.DataFrame({
        "date": dates,
        "% LM-negative": hist_pct_neg,
        "% LM-uncertainty": hist_pct_unc,
        "% LM-positive": hist_pct_pos,
        "Top Bigrams": hist_bigrams
    })

    df_long = hist_df.melt(id_vars="date", var_name="Sentiment", value_name="Percentage")

    chart = (
    alt.Chart(df_long)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:N", axis=alt.Axis(labelAngle=-45, title="Date")),
        y=alt.Y("Percentage:Q", title="Percentage (%)"),
        color=alt.Color("Sentiment:N", title="Sentiment"),  # ← auto colors
        tooltip=["date:N", "Sentiment:N", "Percentage:Q"]
        )
    .properties(title="Language Model Sentiment Over Time", height=400)
    )

    st.altair_chart(chart, use_container_width=True)
    st.subheader("Historical Top Bigrams")
    st.dataframe(hist_df[["date", "Top Bigrams"]])
