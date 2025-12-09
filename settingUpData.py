import matplotlib.pyplot as plt
import loadSession
import pandas as pd
import fastf1
import numpy as np
import math

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
# metrics = ["attacking", "Lap Data"]
columns = pd.MultiIndex.from_product([drivers, metrics])
lapNumbers = sorted(fullLaps["LapNumber"].unique())
lapMatrix = pd.DataFrame(index=lapNumbers, columns=columns)
lapMatrix = lapMatrix.astype(object)

circuitInfo = session.get_circuit_info().corners
circuitInfo['X'] = circuitInfo['X'] / 10
circuitInfo['Y'] = circuitInfo['Y'] / 10


def calculateAngle(point, lapData):
    x1, y1 = lapData.loc[point - 2, ['X', 'Y']]
    x2, y2 = lapData.loc[point - 1, ['X', 'Y']]
    x3, y3 = lapData.loc[point, ['X', 'Y']]
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x3 - x2, y3 - y2])
    normV1 = (abs(v1[0])**2 + abs(v1[1])**2)**0.5
    normV2 = (abs(v2[0])**2 + abs(v2[1])**2)**0.5
    cos_theta = np.dot(v1, v2) / (normV1 * normV2 + 1e-8)
    angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
    return angle


def calculateCornerEntry(lapData, apexIndex, threshold=2):
    entry = apexIndex
    cornerDirection = calculateAngle(entry + 1, lapData)
    while entry > 2:
        if calculateAngle(entry, lapData) - calculateAngle(entry-1, lapData) > threshold:
            break
        entry -= 1
    return entry


def identifyCorner(currentLapData):
    corner_points = np.vstack((circuitInfo['X'].values,
                               circuitInfo['Y'].values)).T
    lap_points = currentLapData[['X', 'Y']].values
    lap = currentLapData.copy()
    lap["Corner"] = False
    lap["Apex"] = False
    apex_indices = []
    corner_entry_indices = []
    for cx, cy in corner_points:
        d = np.sqrt((lap_points[:, 0] - cx)**2 +
                    (lap_points[:, 1] - cy)**2)
        apex_index = int(np.argmin(d))
        apex_indices.append(apex_index)
        entry_index = calculateCornerEntry(lap, apex_index)
        corner_entry_indices.append(entry_index)
    lap.loc[apex_indices, "Apex"] = True
    lap.loc[corner_entry_indices, "Corner"] = True
    return lap


def calculateCornerData(currentLapData):
    cornerIndex = currentLapData.index[currentLapData['Corner'] == True].tolist(
    )
    apexIndex = currentLapData.index[currentLapData['Apex'] == True].tolist()
    distanceToCornerBraking = []
    speedCornerDiff = []
    throttleAtApex = []
    numberOfCorners = len(cornerIndex)
    cornerBrakes = numberOfCorners
    for i in range(numberOfCorners):
        index = cornerIndex[i]
        entry = index
        while entry > 0 and currentLapData.loc[entry, "Speed"] < currentLapData.loc[entry - 1, "Speed"]:
            entry -= 1
        dx = currentLapData.loc[index, "X"] - currentLapData.loc[entry, "X"]
        dy = currentLapData.loc[index, "Y"] - currentLapData.loc[entry, "Y"]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 150:
            distanceToCornerBraking.append(dist)
        elif entry != index:
            cornerBrakes -= 1
        speedCornerDiff.append(
            currentLapData.loc[entry, "Speed"] -
            currentLapData.loc[apexIndex[i], "Speed"]
        )
        throttleAtApex.append(currentLapData.loc[apexIndex[i], "Throttle"])

    if len(distanceToCornerBraking) == 0:
        return None, None, None
    averageDistance = round(sum(distanceToCornerBraking) / cornerBrakes, 2)
    averageSpeedCornerDiff = round(sum(speedCornerDiff) / numberOfCorners, 2)
    averageThrottleAtApex = round(sum(throttleAtApex) / numberOfCorners, 2)
    return averageDistance, averageSpeedCornerDiff, averageThrottleAtApex


def calculatingData(currentLapData):
    totalData = currentLapData.shape[0]
    throttle = currentLapData['Throttle']
    throttleGradient = np.gradient(throttle)
    throttleChange = throttleGradient[throttleGradient != 0]
    throttleOscillation = np.absolute(throttleChange).mean()
    gears = currentLapData['nGear']
    gearGradient = np.gradient(gears)
    diffs = gears.diff()
    upshifts = (diffs > 0).sum()
    downshifts = (diffs < 0).sum()
    gearMean = int(gears.mean())
    gearsCount = gears.value_counts()
    gearPerc = (gearsCount / gearsCount.sum()) * 100
    throttleSD = throttle.std()
    throttleMean = throttle.mean()
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
    brakingPerc = (totalBraking/totalData)*100
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
clearLaps.to_pickle("2025Silverstone2.pkl")
