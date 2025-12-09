import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Russian Passenger Air Service (2007–2020)
    #  Forecasting monthly passengers per airport with Random Forest
    # This notebook:
    # 1. Loads and reshapes the wide-format dataset (months as columns → long format).
    # 2. Builds lag features to capture temporal structure.
    # 3. Trains a RandomForestRegressor to predict monthly passengers.
    # 4. Evaluates model performance (MAE, RMSE).
    # 5. Visualizes predictions vs actual values.
    # 6. Plots feature importances and residual diagnostics.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    import matplotlib.pyplot as plt
    return (
        RandomForestRegressor,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
    )


@app.cell
def _(plt):
    # Make plots a bit bigger
    plt.rcParams["figure.figsize"] = (8, 5)
    return


@app.cell
def _(pd):
    # 1. Load and inspect the data
    df = pd.read_csv("russian_passenger_air_service.csv")
    return (df,)


@app.cell
def _(df):
    # Strip whitespace in column names, just in case
    df_1 = df.rename(columns=lambda x: x.strip())
    print('Columns:', df_1.columns.tolist())
    df_1.head()
    return (df_1,)


@app.cell
def _(df_1):
    # 2. Reshape from wide → long format
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_long = df_1.melt(id_vars=['Airport name', 'Year'], value_vars=months, var_name='month', value_name='passengers')
    df_long = df_long.dropna(subset=['passengers'])
    print('Long-format shape:', df_long.shape)
    # Drop rows with missing passenger counts
    df_long.head()
    return (df_long,)


@app.cell
def _(df_long, pd):
    # 3. Add month numbers and a proper datetime column
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df_long['month_num'] = df_long['month'].map(month_map)
    df_long['date'] = pd.to_datetime(dict(year=df_long['Year'], month=df_long['month_num'], day=1))
    df_long_1 = df_long.sort_values(['Airport name', 'date']).reset_index(drop=True)
    # Build a datetime column at the first day of each month
    # Sort properly
    df_long_1.head()
    return (df_long_1,)


@app.cell
def _(df_long_1):
    # 4. Create lag features per airport
    group = df_long_1.groupby('Airport name')['passengers']
    df_long_1['passengers_lag1'] = group.shift(1)
    df_long_1['passengers_lag2'] = group.shift(2)
    df_long_1['passengers_lag3'] = group.shift(3)
    df_long_2 = df_long_1.dropna(subset=['passengers_lag1', 'passengers_lag2', 'passengers_lag3'])
    # Drop rows where lag values are missing (first 3 months per airport)
    df_long_2.head()
    return (df_long_2,)


@app.cell
def _(cutoff_year, df_long_2, n_lags, pd):
    # 5. Train/test split (time-based)
    cutoff_date = pd.to_datetime(f"{cutoff_year}-01-01")
    train_mask = df_long_2['date'] < cutoff_date
    base_features = ["month_num"]
    lag_features = [
            f"passengers_lag{i}"
            for i in range(1, n_lags + 1)
        ]
    feature_cols = base_features + lag_features
    X = df_long_2[feature_cols]
    y = df_long_2['passengers']
    X_train, y_train = (X[train_mask], y[train_mask])
    X_test, y_test = (X[~train_mask], y[~train_mask])
    print('Train size:', X_train.shape[0])
    print('Test size:', X_test.shape[0])
    return X_test, X_train, feature_cols, train_mask, y_test, y_train


@app.cell
def _(mo):
    lag_select = mo.ui.radio(
            options=["1", "2", "3"],   
            value="3",                 
            label="How many months of history to use"
        )
    lag_select
    return (lag_select,)


@app.cell
def _(lag_select):
    n_lags = int(lag_select.value)
    return (n_lags,)


@app.cell
def _(df_long_2, mo):
    years = df_long_2["date"].dt.year
    min_year = int(years.min())
    max_year = int(years.max())

    cutoff_year_slider = mo.ui.slider(
            start=min_year + 1,          
            stop=max_year - 1,           
            step=1,
            value=2018,                  
            label="Train/test cutoff year"
        )
    cutoff_year_slider
    return (cutoff_year_slider,)


@app.cell
def _(cutoff_year_slider):
    cutoff_year = cutoff_year_slider.value
    return (cutoff_year,)


@app.cell
def _(RandomForestRegressor, X_train, max_depth, y_train, your_var):
    # 6. Train Random Forest model
    model = RandomForestRegressor(
            n_estimators=your_var,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(mo):
    max_depth_slider = mo.ui.slider(
            start=2,
            stop=30,
            step=1,
            value=10,
            label="Max depth of trees"
        )
    unlimited_depth_switch = mo.ui.switch(
            value=False,
            label="No limit on depth"
        )
    mo.hstack([max_depth_slider, unlimited_depth_switch])
    return (max_depth_slider,)


@app.cell
def _(max_depth_slider):
    max_depth = max_depth_slider.value
    return (max_depth,)


@app.cell
def _(mo):
    estimators_slider = mo.ui.slider(start = 10, stop = 800, step = 10, value = 100, label = "Number of trees")
    estimators_slider
    return (estimators_slider,)


@app.cell
def _(estimators_slider):
    your_var = estimators_slider.value
    return (your_var,)


@app.cell
def _(
    X_test,
    mean_absolute_error,
    mean_squared_error,
    model,
    r2_score,
    y_test,
):
    # 7. Evaluate model performance

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:,.4f}")
    return (y_pred,)


@app.cell
def _(plt, y_pred, y_test):
    # 8. Visualization: Predicted vs Actual (scatter plot)

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.4)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual passengers")
    plt.ylabel("Predicted passengers")
    plt.title("Predicted vs Actual passengers (test set)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_long_2, plt, train_mask, y_pred):
    # 9. Visualization: Time series for one example airport
    airport_counts = df_long_2[~train_mask]['Airport name'].value_counts()
    # Choose an airport with many observations in the test set
    example_airport = airport_counts.index[213]
    print('Example airport:', example_airport)
    test_airport_mask = ~train_mask & (df_long_2['Airport name'] == example_airport)
    df_test_airport = df_long_2.loc[test_airport_mask, ['date', 'passengers']].copy()
    df_test_airport['predicted'] = y_pred[test_airport_mask[~train_mask].values]
    df_test_airport = df_test_airport.sort_values('date')
    plt.figure()
    plt.plot(df_test_airport['date'], df_test_airport['passengers'], label='Actual')
    plt.plot(df_test_airport['date'], df_test_airport['predicted'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.title(f'Actual vs Predicted passengers\n{example_airport} (test period)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(feature_cols, model, np, plt):
    # 10. Feature importance plot
    # We inspect which features are most important for the Random Forest.
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importances")
    plt.tight_layout()
    plt.show()

    print("Feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1]):
        print(f"{name:15s}: {imp:.3f}")
    return


@app.cell
def _(plt, y_pred, y_test):
    # 11. Diagnostics: Residual analysis

    residuals = y_test - y_pred

    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual distribution (test set)")
    plt.tight_layout()
    plt.show()

    # Residuals vs predicted
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted passengers")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs predicted (test set)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
