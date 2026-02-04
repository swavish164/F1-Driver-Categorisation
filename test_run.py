import traceback

try:
    from notebook_helpers import build_and_train, predict_lap
    db = r"Databases/database.db"
    print('Using DB:', db)
    model, df = build_and_train(db, year=2025)
    print('Collected rows,cols:', df.shape)
    print('Cluster value counts:')
    print(df['cluster'].value_counts().to_dict())

    test_lap = {
        "throttlePerc100": 48,
        "throttlePerc0": 17,
        "avCornerBrakeDistance": 42,
        "throttleOscillation": 7,
        "coastingPerc": 2.3
    }
    print('Predicting for test lap:', test_lap)
    res = predict_lap(model, test_lap)
    print('Prediction result:')
    print(res)
except Exception as e:
    print('Error during test run:')
    traceback.print_exc()
