import pandas as pd


def cleaningData(session):
    laps = session.laps
    driversData = session._drivers_from_f1_api()
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

    fullGreenLaps = fullLaps[
        (fullLaps['SC'] == False) &
        (fullLaps['VSC'] == False) &
        (fullLaps['Yellow Flag'] == False)
    ]

    driversLaps = fullGreenLaps.groupby("Driver")
    drivers = fullLaps["Driver"].unique()

    metrics = ["attacking", "defending", "Drivers Ahead", "Lap Data","lapId","driverId"]
    columns = pd.MultiIndex.from_product([drivers, metrics])
    lapNumbers = sorted(fullLaps["LapNumber"].unique())

    lapMatrix = pd.DataFrame(index=lapNumbers, columns=columns,dtype=object)
    lapMatrix = lapMatrix.astype(object)

    return {
        'fullLaps': fullLaps,
        'driversLaps': driversLaps,
        'driversData': driversData,
        'drivers': drivers,
        'lapMatrix': lapMatrix,
        'session': session
    }
