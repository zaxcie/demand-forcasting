import datetime
import pandas as pd
import numpy as np


# TODO Could me more precise then / 365 to calculate year elapse
def day_of_year(series, initial_date):
    day_year_with_year_elapse = (series - initial_date).dt.days
    # TODO refactor
    day_year_serie = series.dt.dayofyear
    day_year_serie = day_year_serie.rename("day_year")
    day_year_with_year_elapse = day_year_serie.rename("day_year_with_year_elapse")

    df = pd.concat([day_year_serie, day_year_with_year_elapse],
                   axis=1)

    return df, list(df)


def day_of_month(series, initial_date):
    day_month_serie = series.dt.day
    day_month_serie = day_month_serie.rename("day")
    onehot_day_month_serie = pd.get_dummies(day_month_serie)
    onehot_day_month_serie = onehot_day_month_serie.add_prefix("day_")
    year_elapse = ((series - initial_date).dt.days / 365).apply(np.floor)
    onehot_day_month__counter_year_elapse_serie = onehot_day_month_serie.multiply(year_elapse, axis=0)
    onehot_day_month__counter_year_elapse_serie = onehot_day_month__counter_year_elapse_serie.add_prefix("day_elapse_")

    df = pd.concat([day_month_serie, onehot_day_month_serie, onehot_day_month__counter_year_elapse_serie],
                   axis=1)

    return df, list(df)


def weekday(series):
    df = series.dt.dayofweek
    df = df.rename("weekday")

    return df, [df.name]


def year(series, initial_date):
    year = series.dt.year
    year = year.rename("year")
    year_elapse = ((series - initial_date).dt.days / 365).apply(np.floor)

    onehot_year = pd.get_dummies(year)
    onehot_year = onehot_year.add_prefix("year_")
    onehot_year_elapse = onehot_year.multiply(year_elapse, axis=0)
    onehot_year_elapse = onehot_year_elapse.add_prefix("year_elapse_")

    df = pd.concat([year, onehot_year, onehot_year_elapse], axis=1)

    return df, list(df)


def month(series, initial_date):
    month = series.dt.month
    month = month.rename("month")
    year_elapse = ((series - initial_date).dt.days / 365).apply(np.floor)

    onehot_month = pd.get_dummies(month)
    onehot_month = onehot_month.add_prefix("month_")
    onehot_month_elapse = onehot_month.multiply(year_elapse, axis=0)
    onehot_month_elapse = onehot_month_elapse.add_prefix("month_elapse_")

    df = pd.concat([month, onehot_month, onehot_month_elapse], axis=1)

    return df, list(df)


def week_of_year(series):
    df = series.dt.weekofyear
    df = df.rename("week_of_year")

    return df, [df.name]
