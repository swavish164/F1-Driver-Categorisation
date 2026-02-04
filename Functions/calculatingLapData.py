import numpy as np
from Functions.calculateCornerFunctions import *


def calculatingData(currentLapData, lapId, cursor,session):
    totalData = currentLapData.shape[0]
    throttle = currentLapData['Throttle']
    throttleGradient = np.gradient(throttle)
    throttleChange = throttleGradient[throttleGradient != 0]
    throttleOscillation = np.absolute(throttleChange).mean()

    # gears = currentLapData['nGear']
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

    # brakingCount = braking.value_counts()
    totalThrottle = throttleVC.get(100, 0)
    totalThrottle0 = throttleVC.get(0, 0)
    # totalBraking = brakingCount.get(1)

    throttlePerc = (totalThrottle / totalData) * 100
    throttle0Perc = (totalThrottle0 / totalData) * 100

    avCornerDistance, avSpeedCornerDiff, avApexThrottle = calculateCornerData(
        currentLapData,session
    )

    cursor.execute(
        """INSERT INTO FEATURES (lapId, throttlePerc100, throttlePerc0, avCornerBrakeDistance, throttleOscillation, coastingPerc, brakeChanges, avSpeedCornerDiff, avApexThrottle)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            lapId,
            throttlePerc,
            throttle0Perc,
            avCornerDistance,
            throttleOscillation,
            coastingPerc,
            brakeChanges,
            avSpeedCornerDiff,
            avApexThrottle,
        )
    )

    return {
        "throttlePerc100": throttlePerc,
        "throttlePerc0": throttle0Perc,
        "avCornerBrakeDistance": avCornerDistance,
        "avCornerSpeedDiff": avSpeedCornerDiff,
        "avApexThrottle": avApexThrottle,
        "throttleOscillation": throttleOscillation,
        "brakeChanges": brakeChanges,
        "coastingPerc": coastingPerc
    }