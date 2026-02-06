# Estimation refinement prototype

Adjust location estimates (4×150 kW) by ±1/±2/±3 steps per location and see a portfolio-level score.

- **Data**: Uses a built-in synthetic dataset only (10 locations, no external files).
- **Score**: 100% when refinements are balanced (by base kWh); increases when conservative, decreases when aggressive.
- **Outputs**: Exports and any saved files go only under `prototype/output/`.

## Run the app

From the project root:

```bash
streamlit run prototype/app.py
```

If `streamlit` is not found, use:

```bash
python3 -m streamlit run prototype/app.py
```

Or from `prototype/`:

```bash
streamlit run app.py
# or
python3 -m streamlit run app.py
```

Install dependencies first if needed (use `pip3` if `pip` is not found):

```bash
pip3 install -r prototype/requirements.txt
# or
python3 -m pip install -r prototype/requirements.txt
```
