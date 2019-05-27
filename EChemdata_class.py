import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Cell:

	def __init__(self, sample, directory):
		df = pd.read_csv(f"{directory}/{sample}.csv", index_col=0)
		self.data = df
		# self.cycle = df['File Cycle']
		# self.capacity = df['Capacity/mAh']
		# self.voltage = df['Voltage/V']
		# self.scapacity = df['SCapacity/mAh/g']

	def cycle_data(self, cyclenum):
		"""
			Load the data for a specific cycle, will return a dataframe with same columns as the input dataframe
			Uses "File Cycle" then "Cycle" as the attempted cycle columns
			data is the data series you are interested in
			cycle is the cycle data in an equivalent length array or series
			cycle_num is the cycle number you are interested in

			Returns capacity, voltage
		"""
		# Index of the data corresponding to the cycle of interest
		try:
			idx = self.data['File Cycle'] == cyclenum
		except KeyError:
			idx = self.data['Cycle'] == cyclenum

		return self.data[idx]


