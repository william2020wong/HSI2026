# ==================================================================================
# 專案版本更新紀錄 (Change Log)
# ----------------------------------------------------------------------------------
# 開始日期：2026-08        版本：v1.0    目的：恒指當日高低位預測
# 更新日期：2026-08-30    版本：v2.0    目的：新增 Tab 2 盤中即市動態修正與牛熊證風控指南
# 更新日期：2026-08-30    版本：v2.1    目的：移除固定時間標籤，優化 Tab 2 彈性與數據自動帶入 logic
# 更新日期：2026-08-30    版本：v2.2    目的：修復 Tab 2 視覺對齊，恢復經典 2x2 對稱輸入框矩陣
# ==================================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="恒指波幅與即市戰略系統", page_icon="📈", layout="centered")

st.title("📈 恒指每日波幅與即市戰略系統")
st.caption("隨機森林雙模型 + 時間衰減權重 + 盤中動態牛熊證風控")

# 1. 數據下載與模型訓練 (利用快取加速載入)
@st.cache_data(ttl=3600*4)
def load_data_and_train():
    hsi = yf.download("^HSI", period="1y", interval="1d", progress=False)
    if isinstance(hsi.columns, pd.MultiIndex):
        hsi.columns = hsi.columns.get_level_values(0)

    tr1 = hsi['High'] - hsi['Low']
    tr2 = (hsi['High'] - hsi['Close'].shift(1)).abs()
    tr3 = (hsi['Low'] - hsi['Close'].shift(1)).abs()
    hsi['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
    hsi['ATR_14'] = hsi['TR'].rolling(14).mean()
    hsi['ATR_pct'] = hsi['ATR_14'] / hsi['Open']

    hsi['Parkinson_Vol'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(hsi['High'] / hsi['Low']) ** 2))
    hsi['Upper_Range'] = (hsi['High'] - hsi['Open']) / hsi['Open']
    hsi['Lower_Range'] = (hsi['Open'] - hsi['Low']) / hsi['Open']
    hsi['Is_Bullish'] = (hsi['Close'] > hsi['Open']).astype(int)

    hsi['Upper_Range_D'] = hsi['Upper_Range'].shift(1)
    hsi['Upper_Range_W'] = hsi['Upper_Range'].shift(1).rolling(5).mean()
    hsi['Lower_Range_D'] = hsi['Lower_Range'].shift(1)
    hsi['Lower_Range_W'] = hsi['Lower_Range'].shift(1).rolling(5).mean()
    hsi['ATR_D'] = hsi['ATR_pct'].shift(1)
    hsi['Parkinson_D'] = hsi['Parkinson_Vol'].shift(1)

    df = hsi.dropna().copy()
    features = ['Upper_Range_D', 'Upper_Range_W', 'Lower_Range_D', 'Lower_Range_W', 'ATR_D', 'Parkinson_D']

    n_samples = len(df)
    sample_weights = np.array([0.995 ** (n_samples - i - 1) for i in range(n_samples)])

    rf_upper = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=42)
    rf_lower = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=42)
    rf_class = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

    rf_upper.fit(df[features], df['Upper_Range'], sample_weight=sample_weights)
    rf_lower.fit(df[features], df['Lower_Range'], sample_weight=sample_weights)
    rf_class.fit(df[features], df['Is_Bullish'], sample_weight=sample_weights)

    # 回測計算
    bt_df = df.tail(100).copy()
    bt_pred_upper = rf_upper.predict(bt_df[features])
    bt_pred_lower = rf_lower.predict(bt_df[features])
    bt_pred_high = bt_df['Open'] * (1 + bt_pred_upper)
    bt_pred_low = bt_df['Open'] * (1 - bt_pred_lower)

    mae_high = (bt_df['High'] - bt_pred_high).abs().mean()
    mae_low = (bt_df['Low'] - bt_pred_low).abs().mean()
    both_hit = ((bt_df['High'] <= bt_pred_high) & (bt_df['Low'] >= bt_pred_low)).sum()
    hit_rate = (both_hit / 100) * 100

    bt_pred_class = rf_class.predict(bt_df[features])
    class_acc = (bt_pred_class == bt_df['Is_Bullish']).mean() * 100

    return df, features, rf_upper, rf_lower, rf_class, hit_rate, class_acc, mae_high, mae_low

with st.spinner("正更新數據與模型中..."):
    df, features, rf_upper, rf_lower, rf_class, hit_rate, class_acc, mae_high, mae_low = load_data_and_train()

# 自動掃描權重股業績日
top_stocks = {"0700.HK": "騰訊", "9988.HK": "阿里", "3690.HK": "美團", "0005.HK": "滙豐", "1299.HK": "友邦"}
auto_stage = "0"
detected_events = []
today_date = datetime.now().date()

for ticker, name in top_stocks.items():
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and 'Earnings Date' in cal:
            for ed in cal['Earnings Date']:
                ed_date = ed.date() if isinstance(ed, datetime) else ed
                if ed_date == today_date:
                    detected_events.append(f"💥 {name} 今日發佈業績！")
                    auto_stage = "1"
                elif ed_date == today_date + timedelta(days=1):
                    detected_events.append(f"🤫 {name} 明日發佈業績 (今日觀望)！")
                    if auto_stage != "1": auto_stage = "-1"
    except Exception:
        pass

# 全局基礎數據
last_close = float(df['Close'].iloc[-1])
latest_atr = float(df['ATR_14'].iloc[-1])
latest_date = df.index[-1].strftime('%Y-%m-%d')

latest_input = np.array([[
    df['Upper_Range'].iloc[-1], df['Upper_Range'].iloc[-5:].mean(),
    df['Lower_Range'].iloc[-1], df['Lower_Range'].iloc[-5:].mean(),
    df['ATR_pct'].iloc[-1], df['Parkinson_Vol'].iloc[-1]
]])

base_pred_upper_pct = rf_upper.predict(latest_input)[0]
base_pred_lower_pct = rf_lower.predict(latest_input)[0]
prob_bullish = rf_class.predict_proba(latest_input)[0][1] * 100
prob_bearish = 100 - prob_bullish

# ----------------------------------------------------------------------
# 分頁選單 (Tabs)
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs(["🌅 開市前預測", "⏱️ 盤中即市 & 牛熊戰略"])

# ======================================================================
# Tab 1: 開市前預測 (維持原本最佳介面)
# ======================================================================
with tab1:
    st.subheader("📊 過去 100 日 Backtest 數據")
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("區間涵蓋率", f"{hit_rate:.1f}%")
    col_b2.metric("最高價平均誤差", f"±{mae_high:.1f} 點")
    col_b3.metric("最低價平均誤差", f"±{mae_low:.1f} 點")

    st.divider()

    if detected_events:
        for ev in detected_events: st.warning(ev)

    st.write(f"📅 **數據日期**：{latest_date} | **昨日收市價**：{last_close:.2f} | **14日 ATR**：{latest_atr:.2f} 點")

    night_price = st.number_input("【步驟 1】夜期 / ADR 收市價 (無可留 0)", value=0.0, step=10.0, key="t1_night")
    actual_open = st.number_input("【步驟 2】09:30 實際開市價 (無可留 0)", value=0.0, step=10.0, key="t1_open")

    stage_options = {"0": "☕ 正常交易日 (標準波幅)", "-1": "🤫 消息/業績前夕 (波幅打 8 折)", "1": "💥 消息/業績落地 (波幅放大 1.35倍)"}
    default_idx = 0 if auto_stage == "0" else (1 if auto_stage == "-1" else 2)
    selected_stage = st.selectbox("【步驟 3】交易情境選單", options=list(stage_options.keys()), format_func=lambda x: stage_options[x], index=default_idx, key="t1_stage")

    ref_open = actual_open if actual_open > 0 else (night_price if night_price > 0 else last_close)

    multiplier = 0.80 if selected_stage == "-1" else (1.35 if selected_stage == "1" else 1.00)
    pred_upper_pct = base_pred_upper_pct * multiplier
    pred_lower_pct = base_pred_lower_pct * multiplier

    pred_high = ref_open * (1 + pred_upper_pct)
    pred_low = ref_open * (1 - pred_lower_pct)

    atr_buffer = latest_atr * 0.10
    buy_high, buy_low = pred_low + atr_buffer, pred_low
    sell_low, sell_high = pred_high - atr_buffer, pred_high

    st.divider()
    st.subheader("🎯 今日預測結果")

    col_res1, col_res2 = st.columns(2)
    col_res1.metric("預測最高價 (High)", f"{pred_high:.2f}", f"+{pred_upper_pct*100:.2f}%")
    col_res2.metric("預測最低價 (Low)", f"{pred_low:.2f}", f"-{pred_lower_pct*100:.2f}%")

    st.info(f"📈 陽燭機率：**{prob_bullish:.1f}%** | 📉 陰燭機率：**{prob_bearish:.1f}%** | 全日波幅：**{pred_high - pred_low:.2f} 點**")

    st.subheader("⚔️ Actionable Trading Plan")
    st.write(f"🟢 **多頭低吸區 (Buy Zone)**：`{buy_low:.2f} - {buy_high:.2f}` (止蝕：`{pred_low - 50:.2f}` | 止賺：`{ref_open + (pred_high - ref_open)*0.7:.2f}`)")
    st.write(f"🔴 **空頭高拋區 (Sell Zone)**：`{sell_low:.2f} - {sell_high:.2f}` (止蝕：`{pred_high + 50:.2f}` | 止賺：`{ref_open - (ref_open - pred_low)*0.7:.2f}`)")
    st.write(f"🛡️ **強弱分界線**：`{ref_open:.2f}`")

# ======================================================================
# Tab 2: 盤中即市修正 & 牛熊證風控指南
# ======================================================================
with tab2:
    st.subheader("⏱️ 盤中即市實時動態修正")
    st.caption("輸入開市後實時數據，自動評估剩餘爆發空間與牛熊證收回價安全距離。")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        intraday_open = st.number_input("09:30 開市參考價", value=ref_open, step=10.0, key="t2_open")
        current_price = st.number_input("即時恒指現價", value=ref_open, step=10.0, key="t2_curr")
    with col_i2:
        morning_high = st.number_input("今晨已出現最高價", value=max(ref_open, current_price), step=10.0, key="t2_high")
        morning_low = st.number_input("今晨已出現最低價", value=min(ref_open, current_price), step=10.0, key="t2_low")

    # 動態計算開市前目標
    pred_h_t2 = intraday_open * (1 + base_pred_upper_pct)
    pred_l_t2 = intraday_open * (1 - base_pred_lower_pct)

    # 剩餘波幅計算
    up_space = pred_h_t2 - current_price
    down_space = current_price - pred_l_t2

    # 牛熊證安全距離建議 (1.2 倍 ATR 或最少 380 點)
    safe_distance = max(380, int(latest_atr * 1.2))
    bull_safe_call = current_price - safe_distance
    bear_safe_call = current_price + safe_distance

    st.divider()
    st.subheader("📊 即市動態空間評估")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("距離預測最高價 (衝高空間)", f"{up_space:+.2f} 點")
    col_m2.metric("距離預測最低價 (回落空間)", f"{down_space:+.2f} 點")

    # 交易機會評估邏輯
    st.subheader("💡 即市牛熊證交易決策建議")

    if up_space <= 40 and down_space > 150:
        st.error("⚠️ **現價極度接近預測高位！** 衝高空間嚴重不足，**嚴禁追買牛證**。可關注 Sell Zone 逢高部署熊證阻力。")
    elif down_space <= 40 and up_space > 150:
        st.success("🎯 **現價極度接近預測低位！** 下行空間受限，**具備極佳低吸買牛證盈虧比**。")
    elif up_space > 120 and down_space > 120:
        st.info("⚪ **現價處於中軸震盪區**：上攻與下行均有空間，建議耐心等待價格回落至 Buy Zone 或衝高至 Sell Zone 才出手。")
    else:
        st.warning("⚡ **早盤波動偏大**：請嚴格觀察今晨高低點 (`" + f"{morning_high:.2f}` / `{morning_low:.2f}`" + ") 突破情況。")

    st.divider()
    st.subheader("🛡️ 牛熊證條款選購「防屠房」指南")
    st.write(f"為了防止被大戶盤中「夾波幅 / 暴力拉掃」導致收回，建議選用條款如下：")
    st.write(f"🐂 **牛證建議收回價**：必須 **低於 `{bull_safe_call:.0f}` 點** (距離現價至少 `{safe_distance}` 點安全邊界)")
    st.write(f"🐻 **熊證建議收回價**：必須 **高於 `{bear_safe_call:.0f}` 點** (距離現價至少 `{safe_distance}` 點安全邊界)")
