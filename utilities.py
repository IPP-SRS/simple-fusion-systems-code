import matplotlib.pyplot as plt
import numpy as np


def write_csv(scan, filename):
	"""
	Write scan data in .csv file to import in Mimer.
	"""

	## Write header row
	header_row = ''
	for par_name in scan[0]:
		header_row += par_name + '; '
	header_row += "\n"
	with open(filename, "w") as file:
		file.write(header_row)

	## Write a row for each design point in the scan
	data_rows = ''
	for design_point in scan:
		for par_name in design_point:
			data_rows += str(design_point[par_name]) + '; '
		data_rows += "\n"
	with open(filename, "a") as file:
		file.write(data_rows)


def plot_scan(scan_out, x_par, y_par):
	"""
	Plot scan output
	"""
	X = [scan_out[i][x_par] for i in range(np.size(scan_out))]
	Y = [scan_out[i][y_par] for i in range(np.size(scan_out))]
	plt.scatter(X,Y)
	plt.xlabel(x_par)
	plt.ylabel(y_par)
	plt.savefig('test_plot.png')