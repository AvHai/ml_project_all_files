from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime

app = Flask(__name__)

df = pd.read_csv("flights.csv")

@app.route('/get_flight_by_date', methods=['GET'])
def get_flight_details():
    date_str = request.args.get("date")

    if not date_str:
        return jsonify({"error": "Please provide date"}), 400

    try:
        year = int(date_str[0:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    flights_today = df[
        (df['YEAR'] == year) &
        (df['MONTH'] == month) &
        (df['DAY'] == day)
    ].sort_values(by="SCHEDULED_DEPARTURE").to_dict(orient="records")

    return jsonify({
        "flights_today" : flights_today
    })


if __name__ == '__main__':
    app.run(debug=True)