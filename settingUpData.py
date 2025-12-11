import matplotlib.pyplot as plt
import loadSession
import pandas as pd
import fastf1
import numpy as np
import math
from calculateCornerFunctions import calculateCornerData

fastf1.set_log_level("ERROR")

session = loadSession.session

laps = session.laps
sessionStatus = session.track_status

fullLaps = laps[
    (laps['Sector1Time'] > pd.Timedelta(0)) &
    (laps['Sector2Time'] > pd.Timedelta(0)) &
    (laps['Sector3Time'] > pd.Timedelta(0)) &
    (laps['Deleted'] == False)
].copy()

deletedLaps = laps[laps['Deleted'] == True]

fullLaps['SC'] = False
fullLaps['VSC'] = False
fullLaps['Yellow Flag'] = False

fullLaps['TrackStatus'] = fullLaps['TrackStatus'].fillna('')

sc_mask = fullLaps['TrackStatus'].str.contains('4', regex=False)
vsc_mask = fullLaps['TrackStatus'].str.contains('6', regex=False)
yellow_mask = fullLaps['TrackStatus'].str.contains('2', regex=False)
# red_mask = fullLaps['TrackStatus'].str.contains('5', regex=False)

fullLaps.loc[sc_mask, 'SC'] = True
fullLaps.loc[vsc_mask, 'VSC'] = True
fullLaps.loc[yellow_mask, 'Yellow Flag'] = True
# fullLaps.loc[vsc_mask, 'Red Flag'] = True

fullGreenLaps = fullLaps[(fullLaps['SC'] == False) &
                         (fullLaps['VSC'] == False) &
                         (fullLaps['Yellow Flag'] == False)
                         ]
driversLaps = fullGreenLaps.groupby("Driver")

drivers = fullLaps["Driver"].unique()
metrics = ["attacking", "defending", "Drivers Ahead", "Lap Data"]
columns = pd.MultiIndex.from_product([drivers, metrics])
lapNumbers = sorted(fullLaps["LapNumber"].unique())
lapMatrix = pd.DataFrame(index=lapNumbers, columns=columns)
lapMatrix = lapMatrix.astype(object)


def calculatingData(currentLapData):
    totalData = currentLapData.shape[0]
    throttle = currentLapData['Throttle']
    throttleGradient = np.gradient(throttle)
    throttleChange = throttleGradient[throttleGradient != 0]
    throttleOscillation = np.absolute(throttleChange).mean()
    gears = currentLapData['nGear']
    # gearGradient = np.gradient(gears)
    # diffs = gears.diff()
    # upshifts = (diffs > 0).sum()
    # downshifts = (diffs < 0).sum()
    # gearMean = int(gears.mean())
    # gearsCount = gears.value_counts()
    # gearPerc = (gearsCount / gearsCount.sum()) * 100
    # throttleSD = throttle.std()
    # throttleMean = throttle.mean()
    throttleVC = throttle.value_counts()
    braking = currentLapData['Brake'].astype(int)
    coasting = (throttle < 5) & (braking == 0)
    coastingCount = coasting.sum()
    coastingPerc = (coastingCount / totalData) * 100
    brakeChanges = (braking.diff()).abs().sum()
    brakingCount = braking.value_counts()
    totalThrottle = throttleVC.get(100, 0)
    totalThrottle0 = throttleVC.get(0, 0)
    totalBraking = brakingCount.get(1)
    throttlePerc = (totalThrottle / totalData) * 100
    throttle0Perc = (totalThrottle0 / totalData) * 100
    # brakingPerc = (totalBraking/totalData)*100
    avCornerDistance, avSpeedCornerDiff, avApexThrottle = calculateCornerData(
        currentLapData)

    result = {
        # "upshifts": upshifts,
        # "downshifts": downshifts,
        # "gearMean": gearMean,
        # "gearPerc": gearPerc,
        # "throttleMean": throttleMean,
        # "throttleStd": throttleSD,
        "throttlePerc100": throttlePerc,
        "throttlePerc0": throttle0Perc,
        "avCornerBrakeDistance": avCornerDistance,
        "avCornerSpeedDiff": avSpeedCornerDiff,
        "avApexThrottle": avApexThrottle,
        "throttleOscillation": throttleOscillation,
        "brakeChanges": brakeChanges,
        "coastingPerc": coastingPerc,
    }
    return result


for driver, laps in driversLaps:
    laps = laps.drop(['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                      'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'],
                     axis=1, errors='ignore')
    for i in range(laps.shape[0]):
        lapRow = laps.iloc[i]
        lapNumber = lapRow["LapNumber"]
        lapTelemetry = lapRow.get_telemetry()
        lapTelemetry = lapTelemetry.drop(
            ['Status', 'Z'], axis=1, errors='ignore')
        lapTelemetry['X'] = lapTelemetry['X'] / 10
        lapTelemetry['Y'] = lapTelemetry['Y'] / 10
        lapTelemetry = lapTelemetry.reset_index(drop=True)
        lapTelemetry = identifyCorner(lapTelemetry)
        data = calculatingData(lapTelemetry)
        defendingDrivers = set((lapTelemetry[lapTelemetry['DistanceToDriverAhead'] < 1])[
                               "DriverAhead"].dropna().unique())
        defendingDrivers = [item for item in defendingDrivers]
        if len(defendingDrivers) != 0:
            lapMatrix.loc[lapNumber, (driver, "attacking")] = True
        lapMatrix.loc[lapNumber, (driver, "defending")] = False
        lapMatrix.loc[lapNumber, (driver, "Drivers Ahead")] = defendingDrivers
        lapMatrix.at[lapNumber, (driver, "Lap Data")] = data


for lapNumber in lapMatrix.index:
    for defender in drivers:
        defending = False
        for attacker in drivers:
            lapData = lapMatrix.loc[lapNumber, (attacker, "Lap Data")]
            if lapData is None:
                continue
            data = lapMatrix.loc[lapNumber,
                                 (attacker, "Drivers Ahead")]
            driverNumber = session.get_driver(defender)['DriverNumber']

            if data is None or (isinstance(data, float) and math.isnan(data)):
                driversAhead = []
            elif isinstance(data, list):
                driversAhead = data
            elif isinstance(data, (np.ndarray, pd.Series)):
                driversAhead = list(data)

            if driverNumber in driversAhead:
                defending = True
                break
        lapMatrix.loc[lapNumber, (defender, "defending")] = defending

attackingLaps = lapMatrix.xs("attacking", level=1, axis=1).any(axis=1)
defendingLaps = lapMatrix.xs("defending", level=1, axis=1).any(axis=1)
clearLaps = lapMatrix[~attackingLaps & ~defendingLaps]
attackingLaps = lapMatrix[attackingLaps]
defendingLaps = lapMatrix[defendingLaps]
# clearLaps.to_pickle("2025Silverstone2.pkl")
attackingLaps.to_pickle("attackingLaps2025Silverstone.pkl")
defendingLaps.to_pickle("defendingLaps2025Silverstone.pkl")
