import numpy as np
import loadSession
import math

session = loadSession.session

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
