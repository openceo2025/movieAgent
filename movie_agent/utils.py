import pandas as pd
import streamlit as st


def coerce_int(value, default, label, row_id=None, min_value=None):
    """Safely convert ``value`` to int or return ``default`` with a toast."""
    if pd.isna(value) or str(value).strip() == "":
        return default
    try:
        value = int(value)
    except (ValueError, TypeError):
        if row_id is not None:
            st.toast(f"Invalid {label} for row {row_id}; using default {default}")
        else:
            st.toast(f"Invalid {label}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        if row_id is not None:
            st.toast(
                f"{label.capitalize()} must be >= {min_value} for row {row_id}; using {default}"
            )
        else:
            st.toast(f"{label.capitalize()} must be >= {min_value}; using {default}")
        return default
    return value


def coerce_float(value, default, label, row_id=None, min_value=None):
    """Safely convert ``value`` to float or return ``default`` with a toast."""
    if pd.isna(value) or str(value).strip() == "":
        return default
    try:
        value = float(value)
    except (ValueError, TypeError):
        if row_id is not None:
            st.toast(f"Invalid {label} for row {row_id}; using default {default}")
        else:
            st.toast(f"Invalid {label}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        if row_id is not None:
            st.toast(
                f"{label.capitalize()} must be >= {min_value} for row {row_id}; using {default}"
            )
        else:
            st.toast(f"{label.capitalize()} must be >= {min_value}; using {default}")
        return default
    return value


def rerun_with_message(message: str) -> None:
    """Trigger st.rerun() and show a message after reload."""
    st.session_state["just_rerun"] = message
    st.rerun()
