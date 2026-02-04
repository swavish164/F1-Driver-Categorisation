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
database = sqlite3.connect(
    r"C:\\Users\\swavi\\Documents\\GitHub\\F1-Stop-Strategy\\Databases\\database.db"
)
cursor = database.cursor()

fastf1.set_log_level("ERROR")

cursor.executescript("""
CREATE INDEX IF NOT EXISTS idx_race_year_circuit
ON Race (year, circuit);

CREATE INDEX IF NOT EXISTS idx_driver_code
ON Driver (code);

CREATE INDEX IF NOT EXISTS idx_lap_race_driver
ON LAP (raceId, driverId);

CREATE INDEX IF NOT EXISTS idx_lap_lapId
ON LAP (lapId);

CREATE INDEX IF NOT EXISTS idx_features_lapId
ON Features (lapId);
""")
database.commit()


def calculatingDriverLaps(year, raceNumber, track):
    session = getSession(year, raceNumber)

    cleaned = cleaningData(session)
    driversLaps = cleaned["driversLaps"]
    driversData = cleaned["driversData"]
    drivers = cleaned["drivers"]
    lapMatrix = cleaned["lapMatrix"]

    cursor.execute(
        """INSERT OR IGNORE INTO Race (year, circuit) VALUES (?, ?)""",
        (year, track)
    )
    cursor.execute(
        """SELECT raceId FROM Race WHERE year = ? AND circuit = ?""",
        (year, track)
    )
    raceId = cursor.fetchone()[0]

    driverIdCache = {}

    cornerData = setUpCornerData(session)

    for driver, laps in driversLaps:
        currentDriverData = driversData[
            driversData["Abbreviation"] == driver
        ]

        cursor.execute(
            """
            INSERT OR IGNORE INTO Driver (code, name, team, teamColour)
            VALUES (?, ?, ?, ?)
            """,
            (
                driver,
                str(currentDriverData["FullName"].iloc[0]),
                str(currentDriverData["TeamName"].iloc[0]),
                str(currentDriverData["TeamColor"].iloc[0])
            )
        )

        if driver not in driverIdCache:
            cursor.execute(
                """SELECT driverId FROM Driver WHERE code = ?""",
                (driver,)
            )
            driverIdCache[driver] = cursor.fetchone()[0]

        driverId = driverIdCache[driver]

        laps = laps.drop(
            [
                "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime"
            ],
            axis=1,
            errors="ignore"
        )

        for _, lapRow in laps.iterrows():
            lapNumber = lapRow["LapNumber"]

            cursor.execute(
                """
                INSERT INTO LAP (raceId, driverId, lapNumber, attacking, defending, clean)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (raceId, driverId, lapNumber, 0, 0, 0)
            )
            lapId = cursor.lastrowid

            lapTelemetry = (
                lapRow.get_telemetry()
                .drop(["Status", "Z"], axis=1, errors="ignore")
            )
            lapTelemetry["X"] /= 10
            lapTelemetry["Y"] /= 10
            lapTelemetry = lapTelemetry.reset_index(drop=True)

            lapTelemetry = identifyCorner(lapTelemetry, cornerData)
            calculatingData(lapTelemetry, lapId, cursor, session)

            if defendingDrivers := getDefendingDrivers(lapTelemetry):
                cursor.execute(
                    """UPDATE LAP SET attacking = 1 WHERE lapId = ?""",
                    (lapId,)
                )

                lapMatrix.at[lapNumber, (driver, "defending")] = False
                lapMatrix.at[lapNumber, (driver, "Drivers Ahead")] = defendingDrivers
                lapMatrix.at[lapNumber, (driver, "lapId")] = lapId
                lapMatrix.at[lapNumber, (driver, "driverId")] = driverId

    identifyIfDefending(lapMatrix, drivers, session, cursor, database)

    database.commit()

    lapMatrix = lapMatrix.copy()

    return lapMatrix
