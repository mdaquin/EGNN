import sys
import pandas as pd
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    print("provide a csv file resulting from runing main_gpu.py")
    sys.exit(-1)

res = pd.read_csv(sys.argv[1])
bestperrun = res.groupby("run").MAE.min()
tbestperrun = res.groupby("run").loss.min()
print(f"Average best MAE over 10 runs {bestperrun.mean():.2f}±{bestperrun.std():.2f} on test, {tbestperrun.mean():.2f}±{tbestperrun.std():.2f} on train")


for i in res.run.unique():
    res[res.run == i].set_index("epoch").loss.plot(c=(0.83, 0.20, 0.11, .2))
    res[res.run == i].set_index("epoch").MAE.plot(c=(0.11, 0.20, 0.83, .2))
plt.show()