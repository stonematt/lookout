import streamlit as st
import pandas as pd


def extract_daily_from_field(df, field_name):
    """Extract daily totals from accumulating fields that reset."""
    df_local = df.copy()

    # Convert epoch milliseconds to datetime in Pacific timezone
    df_local["local_datetime"] = pd.to_datetime(
        df_local["dateutc"], unit="ms", utc=True
    ).dt.tz_convert("America/Los_Angeles")
    df_local["local_date"] = df_local["local_datetime"].dt.date

    # Group by date and take max (handles accumulation within period)
    daily_max = df_local.groupby("local_date")[field_name].max()

    # Calculate actual daily totals (handle resets)
    daily_totals = daily_max.diff().fillna(daily_max.iloc[0])
    daily_totals = daily_totals.clip(lower=0)  # Remove negative values from resets

    # Handle edge case where field doesn't reset (use max directly for first day)
    mask = (daily_totals == 0) & (daily_max > 0)
    daily_totals[mask] = daily_max[mask]

    return daily_totals


def extract_hourly_then_sum_to_daily(df):
    """Extract hourly totals, then sum to daily."""
    df_local = df.copy()

    # Keep in UTC for hourly grouping to avoid DST issues
    df_local["utc_datetime"] = pd.to_datetime(df_local["dateutc"], unit="ms", utc=True)
    df_local["utc_hour"] = df_local["utc_datetime"].dt.floor("H")

    # Convert to local time only for date calculation
    df_local["local_datetime"] = df_local["utc_datetime"].dt.tz_convert(
        "America/Los_Angeles"
    )
    df_local["local_date"] = df_local["local_datetime"].dt.date

    # Extract hourly totals using UTC hours
    hourly_max = df_local.groupby("utc_hour")["hourlyrainin"].max()
    hourly_totals = hourly_max.diff().fillna(hourly_max.iloc[0])
    hourly_totals = hourly_totals.clip(lower=0)

    # Handle edge case for first hour
    mask = (hourly_totals == 0) & (hourly_max > 0)
    hourly_totals[mask] = hourly_max[mask]

    # Map UTC hours to local dates for daily aggregation
    hour_to_date = df_local.groupby("utc_hour")["local_date"].first()

    # Sum hourly totals to daily
    hourly_df = pd.DataFrame({"hourly_rain": hourly_totals.values})
    hourly_df["date"] = hourly_totals.index.map(hour_to_date)
    daily_from_hourly = hourly_df.groupby("date")["hourly_rain"].sum()

    return daily_from_hourly


def compare_rain_methods(df):
    """Compare daily rain from dailyrainin vs sum of hourlyrainin."""

    # Method 1: Extract from dailyrainin
    daily_from_daily = extract_daily_from_field(df, "dailyrainin")

    # Method 2: Extract hourly then sum to daily
    daily_from_hourly = extract_hourly_then_sum_to_daily(df)

    # Align dates and compare
    comparison = pd.DataFrame(
        {"date": daily_from_daily.index, "from_daily": daily_from_daily.values}
    )

    hourly_df = pd.DataFrame(
        {"date": daily_from_hourly.index, "from_hourly": daily_from_hourly.values}
    )

    result = comparison.merge(hourly_df, on="date", how="outer").fillna(0)
    result["difference"] = result["from_daily"] - result["from_hourly"]
    result["abs_difference"] = abs(result["difference"])
    result["match"] = result["abs_difference"] < 0.01  # Allow for rounding

    return result


def render():
    """Render the playground tab for testing and development."""
    st.header("Playground 🧪")
    st.write("Development and testing area for new features")

    # Get data from session state
    if "history_df" not in st.session_state:
        st.error("No weather data available in session")
        return

    df = st.session_state["history_df"]

    # Quick validation metrics
    st.subheader("Data Health Check")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        date_range = (df["date"].max() - df["date"].min()).days
        st.metric("Days of Data", date_range)

    with col3:
        current_annual = (
            df["yearlyrainin"].iloc[-1] if "yearlyrainin" in df.columns else 0
        )
        st.metric("Annual Rain (YTD)", f'{current_annual:.2f}"')

    with col4:
        latest_temp = df["tempf"].iloc[-1] if "tempf" in df.columns else 0
        st.metric("Current Temp", f"{latest_temp:.1f}°F")

    # Data range and freshness
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Range:**")
        st.write(f"From: {df['date'].min()}")
        st.write(f"To: {df['date'].max()}")

    with col2:
        st.write("**Rain Fields Available:**")
        rain_fields = [col for col in df.columns if "rain" in col.lower()]
        for field in rain_fields:
            try:
                # Convert to numeric, handling strings and missing values
                numeric_values = pd.to_numeric(df[field], errors="coerce")
                non_zero = (numeric_values > 0).sum()
                total_valid = numeric_values.notna().sum()
                st.write(f"• `{field}`: {non_zero:,} non-zero / {total_valid:,} valid")
            except:
                st.write(f"• `{field}`: Error processing field")

    st.divider()

    st.subheader("Rain Data Validation Test")
    st.write("Comparing daily totals from `dailyrainin` vs sum of `hourlyrainin`")

    if st.button("Run Rain Validation Test"):
        with st.spinner("Running validation..."):
            try:
                results = compare_rain_methods(df)

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", len(results))
                with col2:
                    st.metric("Perfect Matches", results["match"].sum())
                with col3:
                    st.metric("Differences Found", (~results["match"]).sum())

                # Display statistics
                st.write("**Difference Statistics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Max Abs Difference", f"{results['abs_difference'].max():.4f}"
                    )
                with col2:
                    st.metric(
                        "Mean Abs Difference", f"{results['abs_difference'].mean():.4f}"
                    )

                # Show problematic days if any
                if results["abs_difference"].max() > 0.01:
                    st.write("**Days with Significant Differences (>0.01 inches):**")
                    problem_days = results[results["abs_difference"] > 0.01].copy()
                    problem_days = problem_days.sort_values(
                        "abs_difference", ascending=False
                    )
                    display_cols = ["date", "from_daily", "from_hourly", "difference"]
                    st.dataframe(problem_days[display_cols])
                else:
                    st.success("✅ All daily totals match between methods!")

                # Show sample of results
                st.write("**Sample Results:**")
                sample_results = results.head(10)[
                    ["date", "from_daily", "from_hourly", "difference", "match"]
                ]
                st.dataframe(sample_results)

            except Exception as e:
                st.error(f"Error running validation: {e}")

    # Data preview section
    st.subheader("Data Preview")
    if st.checkbox("Show raw data sample"):
        st.write("**Latest 10 records:**")
        preview_cols = [
            "date",
            "dateutc",
            "dailyrainin",
            "hourlyrainin",
            "tempf",
            "humidity",
            "windspeedmph",
        ]
        available_cols = [col for col in preview_cols if col in df.columns]
        st.dataframe(df[available_cols].tail(10))
