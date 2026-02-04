import sqlite3
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

DEFAULT_FEATURE_COLUMNS = [
    "throttlePerc100",
    "throttlePerc0",
    "avCornerBrakeDistance",
    "throttleOscillation",
    "coastingPerc",
]


def connect_db(path: str):
    """Connect to a SQLite database and return a connection."""
    conn = sqlite3.connect(path)
    return conn


def get_races(conn: sqlite3.Connection, year: Optional[int] = None) -> List[int]:
    cur = conn.cursor()
    if year is None:
        cur.execute("SELECT raceID FROM Race")
        return [r[0] for r in cur.fetchall()]
    cur.execute("SELECT raceID FROM Race WHERE year = ?", (year,))
    return [r[0] for r in cur.fetchall()]


def get_drivers_for_race(conn: sqlite3.Connection, race_id: int) -> List[Tuple[int, str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT d.driverId, d.code
        FROM Driver d
        JOIN LAP l ON l.driverId = d.driverId
        WHERE l.raceId = ?
        """,
        (race_id,)
    )
    return cur.fetchall()


def collect_race_data(conn: sqlite3.Connection, race_ids: List[int],
                      feature_columns: Optional[List[str]] = None,
                      averages: bool = True) -> pd.DataFrame:
    """Collect features for the given races.

    If averages is True, returns one row per driver per race with the mean of available laps.
    If averages is False, returns one row per lap with driver and raceId columns.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    rows = []
    cur = conn.cursor()

    for race in race_ids:
        drivers = get_drivers_for_race(conn, race)
        if not drivers:
            continue
        driverIds = [d[0] for d in drivers]
        driverCodes = [d[1] for d in drivers]

        for driverId, driverCode in zip(driverIds, driverCodes):
            cur.execute(
                """
                SELECT lapId
                FROM Lap
                WHERE raceId = ? AND driverId = ? AND attacking = 0 AND defending = 0
                """,
                (race, driverId)
            )
            lapIds = [r[0] for r in cur.fetchall()]

            lapFeatures = []
            for lapId in lapIds:
                cur.execute(
                    f"SELECT {','.join(feature_columns)} FROM Features WHERE lapId = ?",
                    (lapId,)
                )
                row = cur.fetchone()
                if row is None:
                    continue
                if all(x is not None for x in row):
                    lapFeatures.append(tuple(float(x) for x in row))

            if not lapFeatures:
                continue
            lapFeatures = np.array(lapFeatures, dtype=float)
            if averages:
                vals = lapFeatures.mean(axis=0)
                rec = {c: v for c, v in zip(feature_columns, vals)}
                rec.update({"driver": driverCode, "raceId": race})
                rows.append(rec)
            else:
                for r in lapFeatures:
                    rec = {c: v for c, v in zip(feature_columns, r)}
                    rec.update({"driver": driverCode, "raceId": race})
                    rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=feature_columns + ["driver", "raceId"])
    return pd.DataFrame(rows)
