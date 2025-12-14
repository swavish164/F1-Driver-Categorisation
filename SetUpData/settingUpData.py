import matplotlib.pyplot as plt
import fastf1
import numpy as np
import math
import sqlite3
import pandas as pd

from loadSession import getSession
from Functions.calculateCornerFunctions import *
from Functions.identifyDefending import *
from Functions.calculatingLapData import *
from cleaningData import cleaningData

# Connect to database
database = sqlite3.connect(r"C:\\Users\\swavi\\Documents\\GitHub\\F1-Stop-Strategy\\Databases\\database.db")
cursor = database.cursor()

fastf1.set_log_level("ERROR")


def calculatingDriverLaps(year, raceNumber, track):
    session = getSession(year, raceNumber)

    cleaned = cleaningData(session)
    driversLaps = cleaned['driversLaps']
    driversData = cleaned['driversData']
    drivers = cleaned['drivers']
    lapMatrix = cleaned['lapMatrix']

    cursor.execute(
        """INSERT OR IGNORE INTO Race (year, circuit) VALUES (?, ?)""",
        (year, track)
    )
    print("Added race to database")
    database.commit()
    cursor.execute(
        """SELECT raceId FROM Race WHERE year = ? AND circuit = ?""",
        (year, track)
    )
    raceId = cursor.lastrowid

    cornerData = setUpCornerData(session)

    for driver, laps in driversLaps:
        currentDriverData = driversData[driversData['Abbreviation'] == driver]

        cursor.execute(
            """INSERT OR IGNORE INTO Driver (code,name, team) VALUES (?, ?, ?)""",
            (
                driver,
                str(currentDriverData['FullName'].iloc[0]),
                str(currentDriverData['TeamName'].iloc[0])
            )
        )
        print("Added driver"+driver+" to database")
        cursor.execute(
          """SELECT driverId FROM Driver WHERE code = ?""",
          (driver,)
        )
        driverId = cursor.fetchone()[0]
        database.commit()

        laps = laps.drop(
            [
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'
            ],
            axis=1, errors='ignore'
        )

        for i in range(laps.shape[0]):
            lapRow = laps.iloc[i]
            lapNumber = lapRow["LapNumber"]

            cursor.execute(
                """INSERT INTO LAP (raceId, driverId, lapNumber, attacking, defending, clean)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (raceId, driverId, lapNumber, False, False, False)
            )
            print("Added in lap: ",raceId,driverId,lapNumber)
            lapId = cursor.lastrowid
            database.commit()

            lapTelemetry = lapRow.get_telemetry().drop(['Status', 'Z'], axis=1, errors='ignore')
            lapTelemetry['X'] = lapTelemetry['X'] / 10
            lapTelemetry['Y'] = lapTelemetry['Y'] / 10
            lapTelemetry = lapTelemetry.reset_index(drop=True)
            lapTelemetry = identifyCorner(lapTelemetry, cornerData)
            calculatingData(lapTelemetry, lapId, cursor, session)
            defendingDrivers = getDefendingDrivers(lapTelemetry)

            if defendingDrivers:
                cursor.execute(
                    """UPDATE LAP SET attacking=? WHERE driverId=? AND lapId=?""",
                    (1, driverId, lapId)
                )
                print("Set attacking to true for: ",driverId,lapId)
                database.commit()

                lapMatrix.at[lapNumber, (driver, "defending")] = False
                lapMatrix.at[lapNumber, (driver, "Drivers Ahead")] = defendingDrivers
                lapMatrix.at[lapNumber, (driver, "lapId")] = lapId
                lapMatrix.at[lapNumber, (driver, "driverId")] = driverId

    identifyIfDefending(lapMatrix, drivers, session, cursor, database)

    lapMatrix = lapMatrix.copy()

    return lapMatrix