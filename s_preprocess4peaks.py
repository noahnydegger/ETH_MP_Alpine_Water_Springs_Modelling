from pathlib import Path

import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt

def qcol() -> str:
    """Return column name for discharge."""
    return "discharge_(L/min)"


dpath = Path(".") / "Data" / "spring_data" / "wabesense_discharge_2023-09-01"
fpath = dpath / "Oberriet.SS.Ulrika_discharge.csv"
#fpath = dpath / "Bonaduz.BS.Paliu_Fravi_discharge.csv"
#fpath = dpath / "Bonaduz.SS.Leo_Friedrich_discharge.csv"
#fpath = dpath / "Hergiswil.BS.Rossmoos_discharge.csv"
df = pd.read_csv(fpath)

if "a_msg" in locals():
  q = qcol()
else:
  q = "discharge(L/min)"

qs = "discharge_smooth"
# do not forget to fill NA when needed by the algo
df[qs] = df[q].rolling(window=12*60//10).mean().ffill().bfill()

# get peaks

peaks, _ = ss.find_peaks(df[qs], prominence=20, distance=24*60//10)
pw, *_ = ss.peak_widths(df[qs], peaks, rel_height=0.2)


print(pd.DataFrame({"peak_width [h]": pw*10/60}).describe())

fig, ax = plt.subplots()
df[[q, qs]].plot(ax=ax)
ax.plot(peaks, df[qs][peaks], "ob")

fig, ax = plt.subplots()
ax.boxplot(pw * 10 / 60)
ax.set_yscale("log")
ax.set_ylabel("peak width [hr]")
plt.show()


