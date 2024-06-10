import json

from flask import Flask, jsonify

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

from d_first_another import create_date_from_year_week

from d_first_another import initData

app = Flask(__name__)

# Load the dataset
content = initData()


@app.route("/<year>/<month>")
def hello_world(year, month):
    row = {
        'Год': year,
        'Номер недели': month
    }
    # week = year + '-' + month
    return jsonify(execute(row))


def execute(row):
    result = list()
    date = create_date_from_year_week(row)
    for key in content:
        result.append({key['name']: (key['data'][key['data']['Date'] == date])['Work'][0]})

    return result

# content[0]['data'][content[0]['data']['Date'] == create_date_from_year_week(row)]['Work']
