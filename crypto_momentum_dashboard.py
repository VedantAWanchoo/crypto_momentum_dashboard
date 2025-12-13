import streamlit as st

import pandas as pd

import numpy as np

import datetime

from typing import Dict, Tuple, List, Any



# --- HARDCODED GLOBAL DATE CONSTRAINTS ---

GLOBAL_MIN_DATE = datetime.date(2017, 8, 20)

GLOBAL_MAX_DATE = datetime.date(2025, 11, 2)



# --- 1. CONFIGURATION AND DATA LOADING ---

st.set_page_config(

    page_title="Crypto Momentum Backtest Dashboard",

    layout="wide",

    initial_sidebar_state="expanded"

)



st.title("Crypto Momentum Backtest Dashboard")

st.markdown("Adjust the **Time Period** and **Strategy Parameters** in the sidebar, then click **Run**.")



# Dummy paths: You MUST replace these with your actual local file paths.

RETURN_PATH = 'weekly_returns.csv'

VOLUME_PATH = 'weekly_volume.csv'

MCAP_PATH = 'weekly_market_cap.csv'




@st.cache_data

def load_and_process_data(path_return, path_volume, path_mcap):

    """Loads and preprocesses the weekly return, volume, and market cap data."""

    

    try:

        w_return = pd.read_csv(path_return)

        w_volume = pd.read_csv(path_volume)

        w_mcap = pd.read_csv(path_mcap)

    except FileNotFoundError:

        st.error("Data files not found. Please ensure the file paths are correct.")

        st.stop()

    

    def process_df(df):

        df["date"] = pd.to_datetime(df["date"])

        return df.set_index("date").sort_index()



    w_return = process_df(w_return)

    w_volume = process_df(w_volume)

    w_mcap = process_df(w_mcap)

    

    return w_return, w_volume, w_mcap



# Load full data at the start

w_return_full, w_volume_full, w_mcap_full = load_and_process_data(RETURN_PATH, VOLUME_PATH, MCAP_PATH)





# --- 2. BACKTEST UTILITY FUNCTIONS (Your Logic) ---



def cagr(series):

    series = series.dropna()

    if len(series) < 2: return np.nan

    start, end = series.iloc[0], series.iloc[-1]

    years = (series.index[-1] - series.index[0]).days / 365.25

    if years <= 0 or start <= 0: return np.nan

    return (end / start) ** (1 / years) - 1



def ann_std(series):

    return series.dropna().std() * np.sqrt(52)



def max_drawdown(nav):

    nav = nav.dropna()

    roll_max = nav.cummax()

    dd = (nav - roll_max) / roll_max

    return dd.min()


@st.cache_data(show_spinner=False)
def get_market_cap_stats(w_mcap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Market Cap concentration metrics and bottom coin counts."""
    w_mcap = w_mcap.sort_index()
    total_mcap = w_mcap.sum(axis=1)

    # --- 1. Top N Share Concentration ---
    def top_n_share(df, n):
        top_n = df.apply(lambda row: row.sort_values(ascending=False).head(n).sum(), axis=1)
        return (top_n / total_mcap) * 100
        
    share_5 = top_n_share(w_mcap, 5)
    share_10 = top_n_share(w_mcap, 10)
    share_15 = top_n_share(w_mcap, 15)
    share_20 = top_n_share(w_mcap, 20)

    df_concentration = pd.DataFrame({
        "Top 5": share_5,
        "Top 10": share_10,
        "Top 15": share_15,
        "Top 20": share_20
    })

    # --- 2. Bottom 10% Count Metrics ---
    bottom_10_count = []
    total_coins_list = []

    for date, row in w_mcap.iterrows():
        clean = row[row > 0].dropna()
        total_coins = len(clean)

        if total_coins == 0:
            bottom_10_count.append(0)
            total_coins_list.append(0)
            continue

        # Threshold = bottom 10% of total market cap
        thresh = 0.10 * clean.sum()
        sorted_vals = clean.sort_values()
        cumulative = sorted_vals.cumsum()

        # Count coins until cumulative >= 10% of total mcap
        count_bottom = (cumulative <= thresh).sum()

        bottom_10_count.append(count_bottom)
        total_coins_list.append(total_coins)

    df_bottom = pd.DataFrame({
        "bottom_10_count": bottom_10_count,
        "total_coins": total_coins_list
    }, index=w_mcap.index)
    df_bottom["ratio"] = df_bottom["bottom_10_count"] / df_bottom["total_coins"]

    return df_concentration, df_bottom


@st.cache_data(show_spinner=False)
def get_volume_stats(w_volume: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Trading Volume concentration metrics and bottom coin counts."""
    w_vol = w_volume.sort_index()
    total_vol = w_vol.sum(axis=1)

    # --- 1. Top N Share Concentration ---
    def top_n_share_vol(df, n):
        top_n = df.apply(lambda row: row.sort_values(ascending=False).head(n).sum(), axis=1)
        return (top_n / total_vol) * 100

    share_5 = top_n_share_vol(w_vol, 5)
    share_10 = top_n_share_vol(w_vol, 10)
    share_15 = top_n_share_vol(w_vol, 15)
    share_20 = top_n_share_vol(w_vol, 20)

    df_concentration = pd.DataFrame({
        "Top 5": share_5,
        "Top 10": share_10,
        "Top 15": share_15,
        "Top 20": share_20
    })

    # --- 2. Bottom 10% Count Metrics ---
    bottom_10_count = []
    total_coins_list = []

    for date, row in w_vol.iterrows():
        clean = row[row > 0].dropna()
        total_coins = len(clean)

        if total_coins == 0:
            bottom_10_count.append(0)
            total_coins_list.append(0)
            continue

        # Threshold = bottom 10% of total trading volume
        thresh = 0.10 * clean.sum()
        sorted_vals = clean.sort_values()
        cumulative = sorted_vals.cumsum()

        # Count coins until cumulative >= 10% of total volume
        count_bottom = (cumulative <= thresh).sum()

        bottom_10_count.append(count_bottom)
        total_coins_list.append(total_coins)

    df_bottom = pd.DataFrame({
        "bottom_10_count": bottom_10_count,
        "total_coins": total_coins_list
    }, index=w_vol.index)
    df_bottom["ratio"] = df_bottom["bottom_10_count"] / df_bottom["total_coins"]

    return df_concentration, df_bottom



@st.cache_data(show_spinner=False)

def run_backtest(

    data_ret: pd.DataFrame, 

    data_aux: pd.DataFrame, 

    weighting: str, 

    params: Dict[str, Any]

) -> pd.DataFrame:

    """Generic function to run one backtest configuration, handling all weighting/filtering."""

    

    clean_n = params['clean_n']

    lookback_n = params['lookback_n']

    skip_n = params['skip_n']

    holding_n = params['holding_n']

    portfolio_n = params['portfolio_n']

    kill_bottom_filter = params.get('kill_bottom_filter', 0.0)



    add_n = clean_n - lookback_n - skip_n

    total_n = add_n + lookback_n + skip_n + holding_n

    

    required_len = total_n - 1

    if len(data_ret) < required_len:

        return pd.DataFrame()



    results = []



    for i in range(total_n - 1, len(data_ret), holding_n):

        

        ret_window = data_ret.iloc[i - (total_n - 1) : i + 1]

        

        # --- 1. INITIAL UNIVERSE FILTERING ---

        if weighting == 'Equal':

            valid_cols = ret_window.columns[ret_window.notna().all(axis=0)]

            full_window = ret_window[valid_cols]

            aux_window_filtered = None 

        else:

            aux_window = data_aux.loc[ret_window.index]

            common_cols = ret_window.columns.intersection(aux_window.columns)

            ret_sub = ret_window[common_cols]

            aux_sub = aux_window[common_cols]



            mask_ret = ret_sub.notna().all(axis=0)

            mask_aux = aux_sub.notna().all(axis=0) & (aux_sub > 0).all(axis=0)

            valid_cols = common_cols[mask_ret & mask_aux]

            

            full_window = ret_sub[valid_cols]

            aux_window_filtered = aux_sub[valid_cols]

        

        

        # --- 2. KILL BOTTOM FILTER ---

        if kill_bottom_filter > 0.0 and aux_window_filtered is not None and not aux_window_filtered.empty:

            lookback_end_date = ret_window.index[add_n + lookback_n - 1]

            aux_all = aux_window_filtered.loc[lookback_end_date]

            

            aux_sorted = aux_all.sort_values(ascending=True)

            cum_aux = aux_sorted.cumsum()

            

            cutoff = kill_bottom_filter * aux_all.sum()

            to_kill = aux_sorted.index[cum_aux <= cutoff]

            

            valid_cols = [c for c in valid_cols if c not in to_kill]



            if len(valid_cols) == 0: continue

            

            full_window = ret_sub[valid_cols]

            aux_window_filtered = aux_sub[valid_cols]

        

        #if len(valid_cols) < portfolio_n: continue



        # --- 3. MOMENTUM CALCULATION AND SELECTION ---

        lookback_data = full_window.iloc[add_n : add_n + lookback_n]

        mom = (1 + lookback_data).prod() - 1 

        

        mom = mom.dropna()

        #if len(mom) < portfolio_n: continue #valid_cols



        winners = mom.nlargest(portfolio_n).index

        losers = mom.nsmallest(portfolio_n).index

        

        # --- 4. HOLDING PERIOD RETURN CALCULATION ---

        holding_data = full_window.iloc[add_n + lookback_n + skip_n : total_n]



        per_coin_ret = (1 + holding_data[winners]).prod() - 1

        per_coin_ret_losers = (1 + holding_data[losers]).prod() - 1

        per_coin_ret_bench = (1 + holding_data[valid_cols]).prod() - 1

        

        if weighting == 'Equal':

            port_ret = per_coin_ret.mean()

            port_ret_losers = per_coin_ret_losers.mean()

            bench_ret = per_coin_ret_bench.mean()

        

        else: 

            aux_all = aux_window_filtered.loc[lookback_data.index[-1]]

            

            aux_winners = aux_all[winners]

            w_winners = aux_winners / aux_winners.sum()



            aux_losers = aux_all[losers]

            w_losers = aux_losers / aux_losers.sum()



            w_all = aux_all[valid_cols] / aux_all[valid_cols].sum()



            port_ret = (w_winners * per_coin_ret[w_winners.index]).sum()

            port_ret_losers = (w_losers * per_coin_ret_losers[w_losers.index]).sum()

            bench_ret = (w_all * per_coin_ret_bench[w_all.index]).sum()



        block_date = full_window.index[-1]

        

        results.append({

            "Date": block_date,

            "Momentum_Return": port_ret,

            "Reversal_Return": port_ret_losers,

            "Benchmark": bench_ret,

            "Portfolio_N": len(winners),

            "Filtered_Universe_N": len(valid_cols),

        })



    if not results:

        return pd.DataFrame()



    # --- 5. FINAL NAV CALCULATION ---

    momentum_df = pd.DataFrame(results).set_index("Date")

    momentum_df["Momentum_NAV"] = 100 * (1 + momentum_df["Momentum_Return"]).cumprod()

    momentum_df["Reversal_NAV"] = 100 * (1 + momentum_df["Reversal_Return"]).cumprod()

    momentum_df["Benchmark_NAV"] = 100 * (1 + momentum_df["Benchmark"]).cumprod()

    

    return momentum_df





# --- 3. STREAMLIT UI: SIDEBAR INPUTS ---



with st.sidebar:

    st.header(" Time Period Selection")

    

    effective_min_date = max(GLOBAL_MIN_DATE, w_return_full.index.min().date())

    effective_max_date = min(GLOBAL_MAX_DATE, w_return_full.index.max().date())



    start_date = st.date_input(

        "Start Date (Weekly Data)", 

        min_value=effective_min_date, max_value=effective_max_date, 

        value=effective_min_date, 

        help="Selects the first week for the backtest."

    )

    

    end_date = st.date_input(

        "End Date (Weekly Data)", 

        min_value=effective_min_date, max_value=effective_max_date, 

        value=effective_max_date, 

        help="Selects the last week for the backtest."

    )



    if start_date > end_date:

        st.error("Start Date cannot be later than End Date.")

        st.stop()



    st.header(" Strategy Parameters ")

    

    clean_n = st.slider(

        "Universe Clean Lookback (Weeks)", 

        min_value=12, max_value=104, value=52, step=1,

        help="A coin must have clean data for these many prior weeks to be included in the universe."

    )

    

    lookback_n = st.slider(

        "Momentum Lookback (Weeks)", 

        min_value=1, max_value=20, value=4, step=1,

        help="Period used to calculate momentum (e.g., past 4 weeks return)."

    )

    

    skip_n = st.slider(

        "Skip / Lag Period (Weeks)", 

        min_value=0, max_value=10, value=0, step=1,

        help="Weeks between lookback end and holding start."

    )

    

    holding_n = st.slider(

        "Holding Period (Weeks)", 

        min_value=1, max_value=12, value=1, step=1,

        help="How long the portfolio is held before rebalancing."

    )

    

    portfolio_n = st.slider(

        "Portfolio Size (N Coins)", 

        min_value=2, max_value=50, value=10, step=1,

        help="Number of coins in the portfolio."

    )

    

    required_clean_n = lookback_n + skip_n #+ holding_n

    if clean_n < required_clean_n:

        st.error(f"Clean Lookback ({clean_n}) must be ≥ Lookback + Skip + Holding ({required_clean_n}).")

        st.stop()

        

    #st.subheader("Filter Inputs")

    

    kill_bottom_mcap_pct = st.number_input(

        "Kill Bottom Market Cap (%)", 

        min_value=0.0, max_value=10.0, value=2.0, step=0.5, format="%.2f",

        help="Exclude coins in the bottom X% of total market cap at selection date."

    )

    kill_bottom_mcap = kill_bottom_mcap_pct / 100.0

    

    kill_bottom_vol_pct = st.number_input(

        "Kill Bottom Volume (%)", 

        min_value=0.0, max_value=10.0, value=2.0, step=0.5, format="%.2f",

        help="Exclude coins in the bottom X% of total trading volume at selection date."

    )

    kill_bottom_vol = kill_bottom_vol_pct / 100.0

    

    st.markdown("---")

    # --- ADD RUN BUTTON HERE (This places the button in the sidebar) ---

    run_button = st.button("Run")





# --- 4. DATA FILTERING AND CONDITIONAL EXECUTION (MAIN LOGIC) ---



# The entire display logic is contained within this check.

if run_button:

    # Filter the data based on the selected dates

    start_dt = pd.to_datetime(start_date)

    end_dt = pd.to_datetime(end_date)

    

    w_return = w_return_full.loc[start_dt:end_dt]

    w_volume = w_volume_full.loc[start_dt:end_dt]

    w_mcap = w_mcap_full.loc[start_dt:end_dt]



    if w_return.empty or len(w_return) < required_clean_n:

        st.warning(f"The selected time period ({start_date} to {end_date}) is too short or contains insufficient data for the current parameters.")

    

    # 4.1 Define all backtest cases

    backtest_cases = {

        "Equally Weighted": {

            "data_aux": None, 

            "weighting": "Equal", 

            "filter": 0.0

        },

        "Volume Weighted": {

            "data_aux": w_volume, 

            "weighting": "Volume", 

            "filter": 0.0

        },

        "Volume Weighted (with kill filters)": {

            "data_aux": w_volume, 

            "weighting": "Volume", 

            "filter": kill_bottom_vol

        },

        "Market Cap Weighted": {

            "data_aux": w_mcap, 

            "weighting": "Mkt Cap", 

            "filter": 0.0

        },

        "Market Cap Weighted (with kill filters)": {

            "data_aux": w_mcap, 

            "weighting": "Mkt Cap", 

            "filter": kill_bottom_mcap

        },

    }



    # 4.2 Run all backtests and store results

    all_dfs = {}

    common_params = {

        'clean_n': clean_n, 'lookback_n': lookback_n, 

        'skip_n': skip_n, 'holding_n': holding_n, 

        'portfolio_n': portfolio_n

    }



    with st.spinner("Running Backtests..."):

        for name, config in backtest_cases.items():

            params = common_params.copy()

            params['kill_bottom_filter'] = config['filter']

            

            df = run_backtest(

                w_return.copy(), 

                config['data_aux'].copy() if config['data_aux'] is not None else None,

                config['weighting'], 

                params

            )

            all_dfs[name] = df



    st.success("Backtests complete! Review results below.")



    # 4.3 Display results one below the other

    keys = list(all_dfs.keys())

    CUSTOM_COLORS = [ "#1f77b4", "#7f7f7f", "#d62728" ]

    CUSTOM_COLORS_2 = [ "#1f77b4" , "#d62728" ]

    num_total_backtests = len(keys)



#    for name in keys:

 #       momentum_df = all_dfs[name]

    for index, name in enumerate(keys):
  
        momentum_df = all_dfs[name]

        

        # Use a container for each backtest to ensure they stack vertically

        with st.container():
           
            st.markdown("---")
           
            st.markdown(f"## ({index + 1} / {num_total_backtests + 2}) {name}:")



            if momentum_df.empty:

                st.warning("Not enough successful rebalancing points to generate results for this strategy.")

                continue



            # --- Use 3 columns for inner layout (Viz 1, 2, 3 side-by-side) ---

            # Column widths adjusted to give more space to the Summary Table

            col1, col2, col3 = st.columns([1.5, 1, 1])



            # --- COLUMN 1: Performance Summary Table (Visualization 1) ---

            with col1:

                st.markdown("##### 1. Summary Statistics")

                

                time_span_days = (momentum_df.index[-1] - momentum_df.index[0]).days

                

                windows = {

                    "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "7Y": 365 * 7, 

                    "All Time": time_span_days

                }

                strategies = {

                    "Momentum": ("Momentum_NAV", "Momentum_Return"),

                    "Reversal": ("Reversal_NAV", "Reversal_Return"),

                    "Benchmark": ("Benchmark_NAV", "Benchmark")

                }

                rows = []

                

                for label, days in windows.items():

                    

                    if label != "All Time" and time_span_days < days:

                        row = {"Window": label}

                        for strat in strategies:

                            for metric in ["Return", "Risk", "Sharpe", "MDD"]:

                                row[f"{strat}_{metric}"] = np.nan

                        rows.append(row)

                        continue



                    end_date_win = momentum_df.index[-1]

                    start_date_win = momentum_df.index[0] if label == "All Time" else end_date_win - pd.Timedelta(days=days)

                    df_win = momentum_df.loc[momentum_df.index >= start_date_win]



                    row = {"Window": label}

                    

                    for strat, (nav_col, ret_col) in strategies.items():

                        nav = df_win[nav_col].dropna()

                        ret = df_win[ret_col].dropna()

                        

                        if nav.empty or len(ret) < 2 or start_date_win < momentum_df.index[0]:

                            r, s, sharpe_like, mdd = np.nan, np.nan, np.nan, np.nan

                        else:

                            r = cagr(nav)

                            s = ann_std(ret)

                            sharpe_like = r / s if (pd.notna(s) and s != 0) else np.nan

                            mdd = max_drawdown(nav)



                        row[f"{strat}_Return"] = r

                        row[f"{strat}_Risk"] = s

                        row[f"{strat}_Sharpe"] = sharpe_like

                        row[f"{strat}_MDD"] = mdd

                    

                    rows.append(row)

                

                summary = pd.DataFrame(rows).set_index("Window")

                

                # Formatting: Convert NaN (errors/NA) to "-", and percentages/decimals to strings

                summary_fmt = summary.copy()

                for col in summary_fmt.columns:

                    if "Sharpe" in col:

                        summary_fmt[col] = summary_fmt[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

                    else:

                        summary_fmt[col] = summary_fmt[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

                

                mi_cols = [tuple(col.split("_")) for col in summary_fmt.columns]

        

                #summary_fmt.columns = pd.MultiIndex.from_tuples(mi_cols, names=["Strategy", "Metric"])

                summary_fmt.columns = pd.MultiIndex.from_tuples(mi_cols)



                st.dataframe(summary_fmt)



            # --- COLUMN 2: NAV Line Chart (Visualization 2) ---

            with col2:

                st.markdown("##### 2. NAV Comparison (Base: 100)")

                

                nav_data = momentum_df[["Momentum_NAV", "Reversal_NAV", "Benchmark_NAV"]]

                nav_data.columns = ["Momentum", "Reversal", "Benchmark"]

                st.line_chart(nav_data, color=CUSTOM_COLORS)

                

            # --- COLUMN 3: Coin Count Chart (Visualization 3) ---

            with col3:

                st.markdown("##### 3. Coin Count Over Time")

                

                count_data = momentum_df[["Portfolio_N", "Filtered_Universe_N"]]

                count_data.columns = ["Portfolio Size (N)", "Filtered Universe Size"]

                st.line_chart(count_data, color=CUSTOM_COLORS_2)

                

            st.markdown("---") 


    # --- 4.6 MARKET CAP STATS DISPLAY (5/6) ---

    st.header("(6 / 7) Market Capitalization Stats:")
    
    with st.spinner("Calculating Market Cap Stats..."):
        df_mcap_conc, df_mcap_bottom = get_market_cap_stats(w_mcap)

    col_mcap_1, col_mcap_2, col_mcap_3 = st.columns(3)

    # Output 1: Concentration - Title/Y-axis as per request
    with col_mcap_1:
        st.markdown("##### 1. Market Cap of Top N Coins as a % of Total Market Cap")
        st.line_chart(
            df_mcap_conc)
       # st.markdown("_Percentage of Total Market Cap (%)_")

    # Output 2: Bottom 10% Coin Count vs Total
    with col_mcap_2:
        st.markdown("##### 2. Coins (N) that make up bottom 10% of Market Cap vs Total coins")
        df_count = df_mcap_bottom[["bottom_10_count", "total_coins"]].copy()
        df_count.columns = ["Coins making bottom 10% of market cap", "Total coins"]
        st.line_chart(
            df_count)
        #st.markdown("_Number of Coins_")

    # Output 3: Ratio (Bottom 10% Coin Share)
    with col_mcap_3:
        st.markdown("##### 3. Share of Coins (%) that make up bottom 10% of Market Cap")
        df_ratio = df_mcap_bottom[["ratio"]] * 100
        df_ratio.columns = ["Bottom 10% Coin Share"]
        st.line_chart(
            df_ratio)
        #st.markdown("_Percentage_")

    st.markdown("---")


    # --- 4.7 TRADING VOLUME STATS DISPLAY (6/6) ---

    st.header("(7 / 7) Trading Volume Stats:")

    with st.spinner("Calculating Trading Volume Stats..."):
        df_vol_conc, df_vol_bottom = get_volume_stats(w_volume)

    col_vol_1, col_vol_2, col_vol_3 = st.columns(3)

    # Output 1: Concentration (Top N Volume Share) - Title/Y-axis as per request
    with col_vol_1:
        st.markdown("##### 1. Trading Volume of Top N Coins as a % of Total Trading Volume")
        st.line_chart(
            df_vol_conc)
        #st.markdown("_Percentage of Total Trading Volume (%)_")

    # Output 2: Bottom 10% Coin Count vs Total
    with col_vol_2:
        st.markdown("##### 2. Coins (N) that make up bottom 10% of Trading Volume vs Total coins")
        df_count = df_vol_bottom[["bottom_10_count", "total_coins"]].copy()
        df_count.columns = ["Coins making bottom 10% of total trading volume", "Total coins"]
        st.line_chart(df_count)
        #st.markdown("_Number of Coins_")

    # Output 3: Ratio (Bottom 10% Coin Share)
    with col_vol_3:
        st.markdown("##### 3. Share of Coins (%) that make up bottom 10% of Trading Volume")
        df_ratio = df_vol_bottom[["ratio"]] * 100
        df_ratio.columns = ["Bottom 10% Coin Share"]
        st.line_chart(df_ratio)
        #st.markdown("_Percentage_")

    st.markdown("---")


# cd "C:\\Users\\Vedant Wanchoo\\Desktop\\CGS 2020\\Crypto\\CoinDCX Application\\Trial" ; streamlit run crypto_momentum_dashboard.py




