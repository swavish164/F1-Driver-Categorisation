import math
import numpy as np
import pandas as pd


def identifyIfDefending(lapMatrix, drivers, session, cursor, database):
    for lapNumber in lapMatrix.index:
        for defender in drivers:
            driverNumber = session.get_driver(defender)['DriverNumber']
            for attacker in drivers:
                lapData = lapMatrix.at[lapNumber, (attacker, "Lap Data")]
                if lapData is None:
                    continue
                driversAheadData = lapMatrix.at[lapNumber, (attacker, "Drivers Ahead")]
                if isinstance(driversAheadData, list):
                    driversAhead = driversAheadData
                elif isinstance(driversAheadData, (np.ndarray, pd.Series)):
                    driversAhead = list(driversAheadData)
                else:
                    driversAhead = []

                if driverNumber in driversAhead:
                    driverId = lapMatrix.at[lapNumber, (defender, "driverId")]
                    lapId = lapMatrix.at[lapNumber, (defender, "lapId")]

                    cursor.execute(
                        """UPDATE LAP SET defending=? WHERE driverId=? AND lapId=?""",
                        (1, driverId, lapId)
                    )
                    database.commit()


def getDefendingDrivers(lapTelemetry):
    defendingDrivers = set(
        (lapTelemetry[lapTelemetry['DistanceToDriverAhead'] < 1])[
            "DriverAhead"]
        .dropna()
        .unique()
    )
    defending = [item for item in defendingDrivers]
    return defending
