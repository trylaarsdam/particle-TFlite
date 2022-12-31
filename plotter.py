import matplotlib.pyplot as plt

x_data = []
y_data = []

f = open('data.txt','r')

for row in f:
	row = row.split(", ")
	x_data.append(float(row[0]))
	y_data.append(float(row[1]))

plt.scatter(x_data, y_data, color = "red", label = "Linreg output")
plt.xlabel("x", fontsize = 12)
plt.ylabel("y", fontsize = 12)
plt.title("Linreg TFL Output", fontsize = 20)
plt.show()