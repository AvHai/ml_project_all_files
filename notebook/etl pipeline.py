import math, requests, sqlite3
from datetime import date, datetime, timedelta


def extract_flight_data(dt):

    # dt = str(date.fromisoformat("2015-01-01"))
    endpoint=f'http://127.0.0.1:5000/get_flight_by_date?date={dt}'
    response = requests.get(endpoint)

    # Check if request was successful
    if response.status_code == 200:
        return response.json()  # Convert JSON response to Python dict
    else:
        raise Exception(f"Failed to fetch weather data: {response.status_code}")


def transform_flight_data(flight_data):
    flights_today = flight_data['flights_today']
    transformed_data =[]
    for flight in flights_today:
        # print(flight)
        transformed_flight = {
        "UNIQUE_FLIGHT_ID" :    f"{flight['AIRLINE']}{flight['FLIGHT_NUMBER']}",
        "DATE" :                f"{flight['YEAR']}-{flight['MONTH']}-{flight['DAY']}",
        "DAY_OF_WEEK" :         flight['DAY_OF_WEEK'],
        "AIRLINE" :             flight['AIRLINE'],
        "TAIL_NUMBER" :         flight['TAIL_NUMBER'],
        "ORIGIN_AIRPORT" :      flight['ORIGIN_AIRPORT'],
        "DESTINATION_AIRPORT" : flight["DESTINATION_AIRPORT"],
        "SCHEDULED_DEPARTURE" : flight["SCHEDULED_DEPARTURE"],
        "DEPARTURE_TIME" :      flight["DEPARTURE_TIME"],
        "DEPARTURE_DELAY" :     flight["DEPARTURE_DELAY"],
        "TAXI_OUT" :            flight["TAXI_OUT"],
        "WHEELS_OFF" :          flight["WHEELS_OFF"],
        "SCHEDULED_TIME" :      flight["SCHEDULED_TIME"],
        "ELAPSED_TIME" :        flight["ELAPSED_TIME"],
        "AIR_TIME" :            flight["AIR_TIME"],
        "DISTANCE" :            flight["DISTANCE"],
        "WHEELS_ON" :           flight["WHEELS_ON"],
        "TAXI_IN" :             flight["TAXI_IN"],
        "SCHEDULED_ARRIVAL" :   flight["SCHEDULED_ARRIVAL"],
        "ARRIVAL_TIME" :        flight["ARRIVAL_TIME"],
        "ARRIVAL_DELAY" :       flight["ARRIVAL_DELAY"],
        "DIVERTED" :            flight["DIVERTED"],
        "CANCELLED" :           flight["CANCELLED"],
        "CANCELLATION_REASON" : flight["CANCELLATION_REASON"],
        "AIR_SYSTEM_DELAY" :    flight["AIR_SYSTEM_DELAY"],
        "SECURITY_DELAY" :      flight["SECURITY_DELAY"],
        "AIRLINE_DELAY" :       flight["AIRLINE_DELAY"],
        "LATE_AIRCRAFT_DELAY" : flight["LATE_AIRCRAFT_DELAY"],
        "WEATHER_DELAY" :       flight["WEATHER_DELAY"]
        }
        transformed_data.append(transformed_flight)
    return transformed_data


def add_minutes(time_hhmm, minutes):
    # Convert integer HHMM → datetime
    time_str = f"{time_hhmm:04d}"  # Ensure 4 digits (e.g. 5 -> 0005)
    if time_str == "2400":
        time_str = "0000"
    time_obj = datetime.strptime(time_str, "%H%M")
    new_time = time_obj + timedelta(minutes=minutes)
    return int(new_time.strftime("%H%M"))


def get_duration_hhmm(start_hhmm, end_hhmm):
    # Convert to zero-padded strings
    start_str = f"{start_hhmm:04d}"
    end_str = f"{end_hhmm:04d}"

    if start_str == "2400":
        start_str = "0000"
    if end_str == "2400":
        end_str = "0000"

    # Convert to datetime objects (using today's date as base)
    start_time = datetime.strptime(start_str, "%H%M")
    end_time = datetime.strptime(end_str, "%H%M")

    # If end time is before start time, assume it's on the next day
    if end_time < start_time:
        end_time += timedelta(days=1)

    # Return duration in minutes
    return int((end_time - start_time).total_seconds() // 60)



def load_flight_data(transformed_data):
    conn = sqlite3.connect("database.sqlite")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            UNIQUE_FLIGHT_ID TEXT,
            DATE TEXT,
            DAY_OF_WEEK INTEGER,
            AIRLINE TEXT,
            TAIL_NUMBER TEXT,
            ORIGIN_AIRPORT TEXT,
            DESTINATION_AIRPORT TEXT,
            SCHEDULED_DEPARTURE TEXT,
            DEPARTURE_TIME TEXT,
            DEPARTURE_DELAY INTEGER,
            TAXI_OUT INTEGER,
            WHEELS_OFF INTEGER,
            SCHEDULED_TIME INTEGER,
            ELAPSED_TIME INTEGER,
            AIR_TIME INTEGER,
            DISTANCE INTEGER,
            WHEELS_ON INTEGER,
            TAXI_IN INTEGER,
            SCHEDULED_ARRIVAL TEXT,
            ARRIVAL_TIME TEXT,
            ARRIVAL_DELAY INTEGER,
            DIVERTED INTEGER,
            CANCELLED INTEGER,
            CANCELLATION_REASON TEXT,
            AIR_SYSTEM_DELAY INTEGER,
            SECURITY_DELAY INTEGER,
            AIRLINE_DELAY INTEGER,
            LATE_AIRCRAFT_DELAY INTEGER,
            WEATHER_DELAY INTEGER,
            FOREIGN KEY (AIRLINE) REFERENCES airlines(IATA_CODE),
            FOREIGN KEY (ORIGIN_AIRPORT) REFERENCES airports(IATA_CODE),
            FOREIGN KEY (DESTINATION_AIRPORT) REFERENCES airports(IATA_CODE)
    );""")

    for flight in transformed_data:
        print(flight)
        cursor.execute(f"""
            INSERT INTO flights VALUES(
                "{flight["UNIQUE_FLIGHT_ID"]}",
                "{flight["DATE"]}",
                {flight["DAY_OF_WEEK"]},
                "{flight["AIRLINE"]}",
                "{flight["TAIL_NUMBER"] if isinstance(flight['SCHEDULED_DEPARTURE'], str) else "N480HA"}",
                "{flight["ORIGIN_AIRPORT"]}",
                "{flight["DESTINATION_AIRPORT"]}",
                {flight["SCHEDULED_DEPARTURE"]},
                {flight["DEPARTURE_TIME"] if isinstance(flight['DEPARTURE_TIME'], str) else add_minutes(flight["SCHEDULED_DEPARTURE"], 9)},
                {flight["DEPARTURE_DELAY"] if isinstance(flight['DEPARTURE_DELAY'], str) else 9.370158},
                {flight["TAXI_OUT"] if isinstance(flight['TAXI_OUT'], str) else 16.07166},
                {flight["WHEELS_OFF"] if isinstance(flight['WHEELS_OFF'], str) else add_minutes(flight["SCHEDULED_DEPARTURE"], 25)},
                {flight["SCHEDULED_TIME"] if isinstance(flight['SCHEDULED_TIME'], str) else get_duration_hhmm(flight["SCHEDULED_DEPARTURE"], flight["SCHEDULED_ARRIVAL"])},
                {flight["ELAPSED_TIME"] if isinstance(flight['ELAPSED_TIME'], str) else 137.0062},
                {flight["AIR_TIME"] if isinstance(flight['AIR_TIME'], str) else 113.511},
                {flight["DISTANCE"]},
                {flight["WHEELS_ON"] if isinstance(flight['WHEELS_ON'], str) else add_minutes(flight["SCHEDULED_ARRIVAL"], -3)},
                {flight["TAXI_IN"] if isinstance(flight['TAXI_IN'], str) else 7.434971},
                {flight["SCHEDULED_ARRIVAL"]},
                {flight["ARRIVAL_TIME"] if isinstance(flight['ARRIVAL_TIME'], str) else add_minutes(flight["SCHEDULED_ARRIVAL"], 4)},
                {flight["ARRIVAL_DELAY"] if isinstance(flight['ARRIVAL_DELAY'], str) else 4.407057},
                {flight["DIVERTED"] },
                {flight["CANCELLED"] },
                "{flight['CANCELLATION_REASON'] if isinstance(flight['CANCELLATION_REASON'], str) else "B"}",
                {flight["AIR_SYSTEM_DELAY"] if not math.isnan(flight["AIR_SYSTEM_DELAY"]) else 13.48057},
                {flight["SECURITY_DELAY"] if not math.isnan(flight["SECURITY_DELAY"]) else 0.07615387},
                {flight["AIRLINE_DELAY"] if not math.isnan(flight["AIRLINE_DELAY"]) else 18.96955},
                {flight["LATE_AIRCRAFT_DELAY"] if not math.isnan(flight["WEATHER_DELAY"]) else 23.47284},
                {flight["WEATHER_DELAY"] if not math.isnan(flight["WEATHER_DELAY"]) else 2.915290}
            );""")
                
    conn.commit()
    cursor.close()


start_date = date.fromisoformat("2015-03-14")
end_date = date.fromisoformat("2016-01-01")

current_date = start_date
while current_date < end_date:
    flight_data = extract_flight_data(str(current_date))
    transformed_data = transform_flight_data(flight_data)
    load_flight_data(transformed_data)
    current_date += timedelta(days=1)