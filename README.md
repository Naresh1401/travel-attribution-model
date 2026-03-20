# ✈️ Travel Multi-Touch Attribution Model

> **The travel attribution problem:** 47-day research journeys, 3 devices, 8 touchpoints — and 100% of credit goes to the last branded search click.

This project builds a complete multi-touch attribution system for the travel & hospitality industry (OTA / hotel chain), moving beyond simplistic last-click attribution to fairly credit every touchpoint in the notoriously long travel customer journey.

## 🎯 Why This Matters

In travel, customers research for **2–8 weeks** across **multiple devices** before booking. A typical journey:

```
Day 1  📱 Instagram travel inspiration → 
Day 5  📱 Travel blog comparison article → 
Day 12 💻 Google Flights metasearch → 
Day 18 💻 Retargeting display ad → 
Day 25 📧 Email remarketing with deal → 
Day 31 💻 Brand search "Expedia Bali deals" → BOOKING ✈️
```

Under **last-click attribution**, 100% of credit goes to that final brand search. But it was social media, blogs, and metasearch that initiated and nurtured the journey. This project quantifies that gap.

## 🏗️ Project Structure

```
travel-attribution-model/
├── src/
│   ├── generate_data.py       # Synthetic travel journey generator (60K journeys)
│   ├── rule_based.py          # 5 heuristic attribution models
│   ├── markov.py              # Markov chain model with removal effect
│   ├── shapley.py             # Shapley value (game theory) attribution
│   └── booking_window.py      # Travel-specific booking lead time analysis
├── dashboard/
│   └── app.py                 # Streamlit dashboard (5 tabs)
├── data/                      # Generated CSV data (gitignored)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/travel-attribution-model.git
cd travel-attribution-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python -m src.generate_data
```

This creates `data/travel_journeys.csv` with **60,000 synthetic travel journeys** containing:
- 10 marketing channels (paid search, metasearch, social, TV, etc.)
- Cross-device behaviour (mobile → desktop)
- Realistic booking windows (7–56 days of research)
- ~3% conversion rate, avg booking value £650

### 3. Run Attribution Models

```bash
# Rule-based models (last-click, first-click, linear, time-decay, W-shaped)
python -m src.rule_based

# Markov chain attribution
python -m src.markov

# Shapley value attribution
python -m src.shapley

# Booking window analysis
python -m src.booking_window
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## 📊 Attribution Models

### Rule-Based Models

| Model | Description | Travel Insight |
|-------|-------------|----------------|
| **Last-Click** | 100% credit to final touchpoint | Inflates branded search; misses inspiration |
| **First-Click** | 100% credit to first touchpoint | Over-values social & display |
| **Linear** | Equal credit across all touches | Fair but ignores position |
| **Time-Decay** | Exponential decay, 14-day half-life | Suits travel's long research window |
| **W-Shaped** | 30/30/30/10 position-based | Captures travel's 3-phase journey |

### Data-Driven Models

| Model | Description | Travel Insight |
|-------|-------------|----------------|
| **Markov Chain** | Transition probabilities + removal effect | Reveals true channel dependency |
| **Shapley Value** | Game theory — marginal contribution | Fair credit via cooperative game theory |

## 🔑 Key Findings

### Why Metasearch Is the Most Undervalued Channel in Travel

| Channel | Last-Click % | Markov % | Shapley % | Delta |
|---------|-------------|----------|-----------|-------|
| paid_search_brand | ~40% | ~15% | ~13% | **Overvalued** |
| metasearch | ~8% | ~18% | ~16% | **Undervalued** |
| social_inspiration | ~2% | ~12% | ~11% | **Undervalued** |
| tv_brand | ~1% | ~8% | ~9% | **Undervalued** |

> *Exact values depend on the synthetic data seed. Run the models to see precise numbers.*

### Booking Window Insights

- **Last-minute (<7 days):** Converts via direct/brand search with 2-3 touchpoints
- **Short (7-30 days):** Balanced channel mix, 4-5 touchpoints
- **Long (30-90 days):** Needs 3× more social + metasearch touchpoints
- **Early bird (90+ days):** Highest value bookings, begins with TV/social inspiration

### Budget Reallocation Recommendation

Shifting **15% of budget** from branded search to metasearch and social:

| Channel | Current Budget | Recommended | Change |
|---------|---------------|-------------|--------|
| paid_search_brand | 35% | 20% | ↓ -15pp |
| metasearch | 10% | 18% | ↑ +8pp |
| social_inspiration | 5% | 10% | ↑ +5pp |
| tv_brand | 3% | 5% | ↑ +2pp |

**Projected revenue impact:** +5-8% increase in total booking revenue by properly funding upper-funnel channels that initiate high-value journeys.

## 📱 Dashboard

The Streamlit dashboard provides 5 interactive tabs:

1. **Journey Explorer** — Sankey diagram of top 15 converting journey paths
2. **Attribution Comparison** — All 7 models side-by-side per channel
3. **Booking Window** — Channel mix by lead time with stacked bars
4. **Budget Simulator** — Drag sliders → predicted revenue impact
5. **Device Journey** — Cross-device path analysis (mobile → desktop patterns)

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas / numpy** — data processing
- **matplotlib / plotly** — visualisation
- **networkx** — Markov chain graph
- **scikit-learn** — supporting ML utilities
- **Streamlit** — interactive dashboard

## ⚠️ Limitations

1. **Offline touchpoints not captured:** TV, out-of-home (OOH), and word-of-mouth influence bookings but aren't tracked click-by-click. TV brand is modelled as a digital proxy.

2. **Cross-device stitching is imperfect:** We use `customer_id` for deterministic matching. In production, probabilistic matching (device graphs) supplements this but introduces noise.

3. **Synthetic data:** While designed to mirror real OTA patterns, production data will have more noise, seasonal effects, and channel interaction complexity.

4. **No incrementality testing:** Attribution models allocate credit among observed touchpoints but don't prove causation. Geo-based lift tests or randomised holdouts are needed for true incrementality.

5. **Booking value ≠ profit:** Attribution by revenue doesn't account for channel-specific costs (CPA). A channel generating £100K in attributed revenue at 30% CPA is less profitable than one generating £80K at 5% CPA.

## 📚 References

- Shapley, L.S. (1953). *A value for n-person games*
- Anderl, E., et al. (2016). *Mapping the customer journey: Lessons learned from graph-based online attribution modeling*
- Dalessandro, B., et al. (2012). *Causally motivated attribution for online advertising*
- Google (2017). *Data-driven attribution methodology*

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
