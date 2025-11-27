# Streamlit Theme Configuration

## Light Theme Setup

This application uses light theme for better visualization compatibility.

### Local Configuration
The theme is set via `.streamlit/config.toml` (local-only, not tracked in git):

```toml
[theme]
base="light"
```

### Why Local-Only?
- `.streamlit/` directory is in `.gitignore` for security (contains secrets)
- Theme preference is deployment-specific, not code logic
- Allows different environments to choose appropriate themes

### Verification
To verify theme is working:
1. Run app: `streamlit run streamlit_app.py`
2. Check browser dev tools - should show light theme CSS
3. Visualizations should render with proper contrast

### Alternative Approach (if needed)
If config.toml approach doesn't work, theme can be set in code:
```python
# Note: This requires checking valid parameters for your Streamlit version
st.set_page_config(
    page_title="Weather Station Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```