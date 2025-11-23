import loadSession
import pandas as pd
import fastf1

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
    print("Braking: "+str((totalBraking/totalData)*100))
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
            ['Status', 'X', 'Y', 'Z'], axis=1, errors='ignore')
        throttle = calculatingData(lapTelemetry)
        print("Throttle: "+str(throttle))
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
