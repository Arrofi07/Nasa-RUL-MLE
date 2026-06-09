from src.inference.predict import ModelRegistry

registry = ModelRegistry.from_paths()

sample = [
    {
        "engine_id": 1,
        "cycle": 50,
        "setting_1": -0.0007,
        "setting_2": -0.0004,
        "setting_3": 100,
        "sensor_1": 518.67,
        "sensor_2": 641.82,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.02,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_16": 0.03,
        "sensor_17": 392.0,
        "sensor_18": 2388.0,
        "sensor_19": 100.0,
        "sensor_20": 39.06,
        "sensor_21": 23.419,
    }
]

X = registry.pipe.transform_xgb(sample)

print("Expected:", len(registry.xgb.feature_names_in_))
print("Generated:", len(X.columns))

print(set(registry.xgb.feature_names_in_) - set(X.columns))
