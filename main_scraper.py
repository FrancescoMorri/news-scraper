from timeF_scraper import get_today_titles_timeF
from brutalist_scraper import get_today_titles_brutalist, parse_age_to_timedelta
from skimfeed_scraper import get_today_titles_skimfeed
from keyword_frequency import keyword_frequency
from sentiment_tone import add_lm_and_vader, overall_tone_summary, top_lm_terms, load_lm_lexicon_csv

from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import pandas as pd



def get_news_and_analyze():

    full_news = []
    today = datetime.today().strftime('%Y-%m-%d')

    # Get today's business headlines from timeF.com
    items = get_today_titles_timeF()
    if not items:
        print("No business headlines found for today from TimeF.")
    else:
        for it in items:
            full_news.append(it)

    # Get today's business headlines from brutalist.report
    tz = "Europe/Paris"  # change if you want the definition of "today" in another tz
    items = get_today_titles_brutalist(limit=10, today_tz=tz)
    if not items:
        print("No business headlines found for today from brutalist.")
    else:
        for it in items:
            if "age" in it:
                now = datetime.now(ZoneInfo(tz))
                td = parse_age_to_timedelta(it["age"])
                if td:
                    it["date"] = (now - td).isoformat()
            full_news.append(it)

    # Get today's business headlines from skimfeed.com
    data = get_today_titles_skimfeed(only_today_if_possible=True)

    if not data:
        print("No items found with date info. Returning ALL items instead.", file=sys.stderr)
        data = get_today_titles_skimfeed(only_today_if_possible=False)

    if not data:
        print("No items found")
    else:
        for k in data.keys():
            for it in data[k]:
                standardized_it = {
                    "title": it.title,
                    "url": it.url,
                    "date": it.published,
                    "source": it.section if it.section != 'Latest' else '',
                }
                full_news.append(standardized_it)

    store_data = {}
    store_data['date'] = today

    # Keyword frequency analysis
    keyword_freq_results = keyword_frequency(full_news, top_n=30)

    # get top N unigrams and bigrams
    n = 10
    df = keyword_freq_results['top_unigrams']
    top_unigrams = (df.sort_values("count", ascending=False)
                 .head(n)
                 .loc[:, ["term", "count"]])
    store_data['top_unigrams'] = top_unigrams.to_dict(orient="records")
    df = keyword_freq_results['top_bigrams']
    top_bigrams = (df.sort_values("count", ascending=False)
                 .head(n)
                 .loc[:, ["term", "count"]])
    store_data['top_bigrams'] = top_bigrams.to_dict(orient="records")

    # Sentiment/Tone analysis
    titles_only_df = pd.DataFrame(full_news, columns=["title"])

    scored = add_lm_and_vader(titles_only_df, "Loughran-McDonald_MasterDictionary_1993-2024.csv")
    most_negative = scored.loc[scored["lm_neg"].idxmax()]
    store_data['most_negative'] = most_negative['title']
    store_data['most_negative_lm_score'] = most_negative['lm_neg']
    store_data['most_negative_vader_score'] = most_negative['vader_compound']
    most_positive = scored.loc[scored["lm_pos"].idxmax()]
    store_data['most_positive'] = most_positive['title']
    store_data['most_positive_lm_score'] = most_positive['lm_pos']
    store_data['most_positive_vader_score'] = most_positive['vader_compound']
    most_uncertain = scored.loc[scored["lm_uncert"].idxmax()]
    store_data['most_uncertain'] = most_uncertain['title']
    store_data['most_uncertain_lm_score'] = most_uncertain['lm_uncert']
    store_data['most_uncertain_vader_score'] = most_uncertain['vader_compound']
    most_litigious = scored.loc[scored["lm_litig"].idxmax()]
    store_data['most_litigious'] = most_litigious['title']
    store_data['most_litigious_lm_score'] = most_litigious['lm_litig']
    store_data['most_litigious_vader_score'] = most_litigious['vader_compound']
    most_constraining = scored.loc[scored["lm_constr"].idxmax()]
    store_data['most_constraining'] = most_constraining['title']
    store_data['most_constraining_lm_score'] = most_constraining['lm_constr']
    store_data['most_constraining_vader_score'] = most_constraining['vader_compound']

    # Distributions
    vader_mean = float(scored["vader_compound"].mean())
    # number of headlines with neg/unc/pos tokens
    pct_neg    = float(scored["lm_has_neg"].mean()) * 100.0
    pct_unc    = float(scored["lm_has_uncert"].mean()) * 100.0
    pct_pos    = float(scored["lm_has_pos"].mean()) * 100.0
    store_data['vader_mean'] = vader_mean
    store_data['head_w_neg'] = pct_neg
    store_data['head_w_unc'] = pct_unc
    store_data['head_w_pos'] = pct_pos

    # shares of neg/pos/unc/litig/constr tokens
    negative_pct = float(scored["lm_neg_share"].mean())
    positive_pct = float(scored["lm_pos_share"].mean())
    uncertainty_pct = float(scored["lm_uncert_share"].mean())
    litigious_pct = float(scored["lm_litig_share"].mean())
    constraining_pct = float(scored["lm_constr_share"].mean())
    store_data['neg_share_of_tok'] = negative_pct
    store_data['pos_share_of_tok'] = positive_pct
    store_data['unc_share_of_tok'] = uncertainty_pct
    store_data['litig_share_of_tok'] = litigious_pct
    store_data['constr_share_of_tok'] = constraining_pct

    # Overall tone
    #tone_df = overall_tone_summary(scored)

    # Top LM terms
    tops = top_lm_terms(scored, load_lm_lexicon_csv("Loughran-McDonald_MasterDictionary_1993-2024.csv"))
    store_data['top_lm_negative'] = tops['negative'].head(10).to_dict()
    store_data['top_lm_positive'] = tops['positive'].head(10).to_dict()
    

    return store_data
