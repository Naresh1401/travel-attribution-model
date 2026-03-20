"""
Synthetic Travel Journey Data Generator
========================================
Generates 60,000 realistic travel customer journeys with multi-device,
multi-channel behaviour typical of OTA / hotel booking research.

Journey characteristics:
- 1-12 touchpoints spanning 7-56 days of research
- Mobile = inspiration (social, blogs) → Desktop = booking (paid search, direct)
- Metasearch critical mid-funnel; branded search near conversion
- ~3% overall conversion rate; avg booking value £650
"""

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NUM_JOURNEYS = 60_000

CHANNELS = [
    "paid_search_brand",
    "paid_search_generic",
    "metasearch",
    "display_retargeting",
    "email_remarketing",
    "social_inspiration",
    "travel_blog_affiliate",
    "direct",
    "organic_search",
    "tv_brand",
]

DEVICES = ["mobile", "tablet", "desktop"]

DESTINATION_CATEGORIES = [
    "beach_resort",
    "city_break",
    "ski_holiday",
    "cruise",
    "adventure_tour",
    "luxury_resort",
]

TRIP_TYPES = ["leisure", "business"]


def _generate_customer_id(idx: int) -> str:
    """Deterministic customer ID from index."""
    return hashlib.sha256(f"customer_{idx}".encode()).hexdigest()[:12]


def _pick_journey_length(rng: np.random.Generator) -> int:
    """Weighted journey length: mostly 2-6 touches, fewer 1 or 7-12."""
    weights = np.array([0.08, 0.15, 0.20, 0.20, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01])
    return rng.choice(np.arange(1, 13), p=weights)


def _pick_channel_sequence(n_touches: int, rng: np.random.Generator, will_convert: bool):
    """
    Generate a realistic channel sequence.

    Awareness → Consideration → Conversion funnel:
      Early: social_inspiration, tv_brand, travel_blog_affiliate, display_retargeting
      Mid:   metasearch, organic_search, paid_search_generic, email_remarketing
      Late:  paid_search_brand, direct, metasearch
    """
    early_channels = ["social_inspiration", "tv_brand", "travel_blog_affiliate", "display_retargeting"]
    mid_channels = ["metasearch", "organic_search", "paid_search_generic", "email_remarketing"]
    late_channels = ["paid_search_brand", "direct", "metasearch"]

    early_w = np.array([0.30, 0.15, 0.25, 0.30])
    mid_w = np.array([0.35, 0.25, 0.20, 0.20])
    late_w = np.array([0.45, 0.35, 0.20])

    sequence = []
    for i in range(n_touches):
        position = i / max(n_touches - 1, 1)  # 0.0 → 1.0

        if position < 0.33:
            pool, weights = early_channels, early_w
        elif position < 0.66:
            pool, weights = mid_channels, mid_w
        else:
            pool, weights = late_channels, late_w

        # Add some noise: 20% chance to pick from any stage
        if rng.random() < 0.20:
            ch = rng.choice(CHANNELS)
        else:
            ch = rng.choice(pool, p=weights)
        sequence.append(ch)

    # If converting, strongly bias the last touch toward brand search / direct
    if will_convert and n_touches >= 2:
        last_touch_pool = ["paid_search_brand", "direct", "metasearch"]
        last_touch_w = np.array([0.50, 0.30, 0.20])
        sequence[-1] = rng.choice(last_touch_pool, p=last_touch_w)

    return sequence


def _pick_device_sequence(channels: list[str], rng: np.random.Generator):
    """
    Cross-device behaviour:
    - Inspiration channels (social, blogs, tv) → mostly mobile
    - Mid-funnel → mix of mobile & desktop
    - Conversion (brand search, direct) → mostly desktop
    """
    mobile_heavy = {"social_inspiration", "tv_brand", "travel_blog_affiliate"}
    desktop_heavy = {"paid_search_brand", "direct"}

    devices = []
    for ch in channels:
        if ch in mobile_heavy:
            d = rng.choice(DEVICES, p=[0.65, 0.15, 0.20])
        elif ch in desktop_heavy:
            d = rng.choice(DEVICES, p=[0.15, 0.10, 0.75])
        else:
            d = rng.choice(DEVICES, p=[0.35, 0.15, 0.50])
        devices.append(d)
    return devices


def generate_journeys(n_journeys: int = NUM_JOURNEYS, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic travel journey data."""
    rng = np.random.Generator(np.random.PCG64(seed))

    # Pre-decide conversion: ~3% conversion rate
    conversion_flags = rng.random(n_journeys) < 0.03

    rows = []

    for idx in range(n_journeys):
        customer_id = _generate_customer_id(idx)
        will_convert = bool(conversion_flags[idx])

        n_touches = _pick_journey_length(rng)
        # Converters tend to have more touchpoints
        if will_convert:
            n_touches = max(n_touches, rng.integers(3, 10))

        channels = _pick_channel_sequence(n_touches, rng, will_convert)
        devices = _pick_device_sequence(channels, rng)

        # Journey span: varied range to create all booking window segments
        # 10% last-minute (0-6 days), 35% short (7-30), 35% long (30-90), 20% early (90+)
        span_bucket = rng.random()
        if span_bucket < 0.10:
            journey_span_days = int(rng.integers(0, 7))
        elif span_bucket < 0.45:
            journey_span_days = int(rng.integers(7, 31))
        elif span_bucket < 0.80:
            journey_span_days = int(rng.integers(30, 91))
        else:
            journey_span_days = int(rng.integers(90, 151))
        if n_touches == 1:
            journey_span_days = min(journey_span_days, 3)  # single-touch: keep short

        # Base timestamp: random start within 2024
        base_ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))

        # Distribute touchpoints across the journey span
        if n_touches > 1:
            offsets_days = np.sort(rng.uniform(0, journey_span_days, n_touches))
            offsets_days[0] = 0.0
            offsets_days[-1] = float(journey_span_days)
        else:
            offsets_days = np.array([0.0])

        # Trip metadata (same for entire journey)
        dest = rng.choice(DESTINATION_CATEGORIES)
        trip = rng.choice(TRIP_TYPES, p=[0.75, 0.25])

        # Days before departure: 1-180
        days_before_departure = int(rng.integers(1, 181))

        # Booking value for converters
        if will_convert:
            if dest in ("luxury_resort", "cruise"):
                bv = rng.normal(1200, 300)
            elif dest == "city_break":
                bv = rng.normal(450, 120)
            elif trip == "business":
                bv = rng.normal(550, 150)
            else:
                bv = rng.normal(650, 200)
            booking_value = max(round(float(bv), 2), 50.0)
        else:
            booking_value = 0.0

        booking_lead_time = journey_span_days

        for t in range(n_touches):
            session_ts = base_ts + pd.Timedelta(days=float(offsets_days[t]))
            # Add intra-day randomness
            session_ts += pd.Timedelta(hours=int(rng.integers(6, 24)),
                                       minutes=int(rng.integers(0, 60)))
            session_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{customer_id}_{t}_{idx}").hex[:16]

            rows.append({
                "customer_id": customer_id,
                "session_id": session_id,
                "session_timestamp": session_ts,
                "channel": channels[t],
                "device": devices[t],
                "converted": int(will_convert) if t == n_touches - 1 else 0,
                "booking_value": booking_value if (will_convert and t == n_touches - 1) else 0.0,
                "destination_category": dest,
                "trip_type": trip,
                "days_before_departure": days_before_departure,
                "booking_lead_time": booking_lead_time,
                "touchpoint_number": t + 1,
                "total_touchpoints": n_touches,
                "journey_id": f"J{idx:06d}",
            })

    df = pd.DataFrame(rows)
    df["session_timestamp"] = pd.to_datetime(df["session_timestamp"])
    df = df.sort_values(["customer_id", "session_timestamp"]).reset_index(drop=True)
    return df


def save_data(df: pd.DataFrame, output_dir: str = "data") -> Path:
    """Save generated data to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "travel_journeys.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} touchpoint rows ({df['journey_id'].nunique():,} journeys) → {path}")
    return path


if __name__ == "__main__":
    df = generate_journeys()
    save_data(df)

    # Quick summary
    print(f"\n--- Data Summary ---")
    print(f"Total touchpoints:  {len(df):,}")
    print(f"Total journeys:     {df['journey_id'].nunique():,}")
    converters = df.groupby("journey_id")["converted"].max()
    print(f"Conversions:        {converters.sum():,} ({converters.mean():.1%})")
    print(f"Avg booking value:  £{df.loc[df['booking_value'] > 0, 'booking_value'].mean():.0f}")
    print(f"Channels:           {df['channel'].nunique()}")
    print(f"Avg touchpoints:    {df.groupby('journey_id').size().mean():.1f}")
    print(f"\nChannel distribution:\n{df['channel'].value_counts().to_string()}")
    print(f"\nDevice distribution:\n{df['device'].value_counts().to_string()}")
