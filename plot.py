import matplotlib.pyplot as plt
from psogwo import HPSOGWO
import benchmarks


function_to_be_used = benchmarks.F5


d = HPSOGWO(function_to_be_used)
d.opt()
x2 = d.return_result()


# PLOT
plt.figure()

plt.plot(x2)

plt.grid()
plt.legend(["HPSOGWO"], loc="upper right")
plt.title("Comparision of HPSOGWO with PSO and GWO")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Score")
plt.show()