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
columns = pd.MultiIndex.from_product([drivers, metrics])
lapNumbers = sorted(fullLaps["LapNumber"].unique())
lapMatrix = pd.DataFrame(index=lapNumbers, columns=columns)
lapMatrix = lapMatrix.astype(object)

circuitInfo = session.get_circuit_info().corners


def calculateCornerEntry(lapData, apexIndex, threshold=5):
    entry = apexIndex
    while entry > 1:
        x1, y1 = lapData.loc[entry - 2, ['X', 'Y']]
        x2, y2 = lapData.loc[entry - 1, ['X', 'Y']]
        x3, y3 = lapData.loc[entry, ['X', 'Y']]
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])
        normV1 = (abs(v1[0])**2 + abs(v1[1])**2)**0.5
        normV2 = (abs(v2[0])**2 + abs(v2[1])**2)**0.5
        cos_theta = np.dot(v1, v2) / (normV1 * normV2 + 1e-8)
        angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
        if angle > threshold:
            break
        entry -= 1
    return entry


def identifyCorner(currentLapData):
    cornerPoints = pd.DataFrame({
        "X": circuitInfo['X'].values,
        "Y": circuitInfo['Y'].values,
        "CornerMarker": True,
        "Apex": True
    })
    lap = currentLapData.copy()
    lap["CornerMarker"] = False
    lap["Apex"] = False
    lap["Corner"] = False
    combined = pd.concat([lap, cornerPoints], ignore_index=True)
    combined = combined.sort_values("X").reset_index(drop=True)
    apexIndices = combined.index[combined["Apex"] == True].tolist()
    cornerDataIndices = []
    for index in apexIndices:
        prev = index - 1
        next = 1 + index
        if prev >= 0 and not combined.loc[prev, "CornerMarker"]:
            marker = combined.loc[index]['X']
            previousPoint = marker - combined.loc[prev]['X']
            nextPoint = marker - combined.loc[next]['X']
            if previousPoint <= nextPoint:
                apexIndices.append(prev)
            else:
                apexIndices.append(next)
            cornerDataIndices.append(
                calculateCornerEntry(combined, index))

            combined.loc[next, "Apex"] = True
    combined.loc[cornerDataIndices, "Corner"] = True
    combined.loc[apexIndices, "Apex"] = True
    lapWithCorners = combined[combined["CornerMarker"] == False].copy()
    lapWithCorners = lapWithCorners.drop(columns=["CornerMarker"])
    lapWithCorners = lapWithCorners.reset_index(drop=True)
    return lapWithCorners


def calculateCornerData(currentLapData):
    cornerIndex = currentLapData.index[currentLapData['Corner'] == True].tolist(
    )
    apexIndex = currentLapData.index[currentLapData['Apex'] == True].tolist()
    print(cornerIndex, apexIndex)
    distanceToCornerBraking = []
    speedCornerDiff = []
    throttleAtApex = []
    numberOfCorners = len(cornerIndex)
    for i in range(numberOfCorners):
        index = cornerIndex[i]
        entry = index
        while entry > 0 and currentLapData.loc[entry, "Speed"] >= currentLapData.loc[entry - 1, "Speed"]:
            entry -= 1
        # print("Index: "+str(index) + " Entry: "+str(entry))
        dx = currentLapData.loc[index, "X"] - currentLapData.loc[entry, "X"]
        dy = currentLapData.loc[index, "Y"] - currentLapData.loc[entry, "Y"]
        dist = math.sqrt(dx*dx + dy*dy)
        distanceToCornerBraking.append(dist)
        speedCornerDiff.append(
            currentLapData.loc[entry, "Speed"] -
            currentLapData.loc[apexIndex[i], "Speed"]
        )
        throttleAtApex.append(currentLapData.loc[apexIndex[i], "Throttle"])

    if len(distanceToCornerBraking) == 0:
        return None, None, None
    averageDistance = round(sum(distanceToCornerBraking) / numberOfCorners, 2)
    averageSpeedCornerDiff = round(sum(speedCornerDiff) / numberOfCorners, 2)
    averageThrottleAtApex = round(sum(throttleAtApex) / numberOfCorners, 2)
    print("Average distance: "+str(averageDistance))
    print("Average speed dif: "+str(averageSpeedCornerDiff))
    print("Average apex throttle: "+str(averageThrottleAtApex)+"\n")
    return averageDistance, averageSpeedCornerDiff, averageThrottleAtApex


def calculatingData(currentLapData):
    totalData = currentLapData.shape[0]
    throttle = currentLapData['Throttle']
    gears = currentLapData['nGear']
    diffs = gears.diff()
    upshifts = (diffs > 0).sum()
    downshifts = (diffs < 0).sum()
    gearMean = int(gears.mean())
    gearsCount = gears.value_counts()
    gearPerc = (gearsCount / gearsCount.sum()) * 100
    throttleSD = throttle.std()
    throttleMean = throttle.mean()
    throttleVC = throttle.value_counts()
    braking = currentLapData['Brake'].value_counts()
    totalThrottle = throttleVC.get(100, 0)
    totalThrottle0 = throttleVC.get(0, 0)
    totalBraking = braking.get(True)
    throttlePerc = (totalThrottle / totalData) * 100
    brakingPerc = (totalBraking/totalData)*100
    avCornerDistance, avSpeedCornerDiff, avApexThrottle = calculateCornerData(
        currentLapData)

    return ((totalThrottle / totalData) * 100)


for driver, laps in driversLaps:
    laps = laps.drop(['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                      'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'],
                     axis=1, errors='ignore')
    for i in range(1):
        lapRow = laps.iloc[i]
        lapNumber = lapRow["LapNumber"]
        lapTelemetry = lapRow.get_telemetry()
        lapTelemetry = lapTelemetry.drop(
            ['Status', 'Z'], axis=1, errors='ignore')
        lapTelemetry = lapTelemetry.reset_index(drop=True)
        lapTelemetry = identifyCorner(lapTelemetry)
        throttle = calculatingData(lapTelemetry)
        minDistanceToDriverAhead = lapTelemetry["DistanceToDriverAhead"].min()
        driversAhead = set(lapTelemetry["DriverAhead"].dropna().unique())
        attacking = minDistanceToDriverAhead < 1
        defendingDriver = lapTelemetry["DriverAhead"] if attacking else 0

        lapMatrix.loc[lapNumber, (driver, "attacking")] = attacking
        lapMatrix.loc[lapNumber, (driver, "defending")] = False
        lapMatrix.loc[lapNumber, (driver, "Drivers Ahead")] = driversAhead
        lapMatrix.at[lapNumber, (driver, "Lap Data")] = lapTelemetry

for lapNumber in lapMatrix.index:
    for defender in drivers:
        defending = False
        for attacker in drivers:
            lapData = lapMatrix.loc[lapNumber, (attacker, "Lap Data")]
            if lapData is None:
                continue
            driversAhead = lapMatrix.loc[lapNumber,
                                         (attacker, "Drivers Ahead")]
            driverNumber = session.get_driver(defender)['DriverNumber']
            if not pd.isna(driversAhead) and driverNumber in driversAhead:
                defending = True
                break
        lapMatrix.loc[lapNumber, (defender, "defending")] = defending

# lapMatrix.to_pickle("2025Silverstone.pkl")
