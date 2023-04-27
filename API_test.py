import water_level_pre
import read_data as rd


data, max, min = rd.read()
re = water_level_pre.predict_water_level(data, max, min)

print(re)
