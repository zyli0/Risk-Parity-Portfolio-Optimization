import numpy as np
import pandas as pd


class YieldByFrequency:

    def process(self, dataframe, vtype):
        ar = np.zeros((int((dataset.shape[0]) / 5) + 1))
        for j in range(len(dataset.columns)):
            parray = np.array([])
            i = 0
            while i <= dataset.shape[0] - div:
                wyield = 1 + dataset.iloc[i, j]
                for count in range(1, div):
                    i += 1
                    wyield = wyield * (1 + dataset.iloc[i, j])
                    count += 1
                parray = np.append(parray, wyield)
                i += 1

            numleft = dataset.shape[0] - i
            wyield = 1 + dataset.iloc[i, j]

            while numleft > 1:
                i += 1
                wyield = wyield * (1 + dataset.iloc[i, j])
                numleft -= 1
            parray = np.append(parray, wyield)

            ar = np.column_stack((ar, parray))

            j += 1

        ar = np.delete(ar, 0, 1)
        newframe = pd.DataFrame(ar, columns=["A", "B", "C", "D", "E"])
        print(newframe)


dataset = pd.read_excel('/Users/lizongyun/Desktop/中信建投/投研/Test.xls', sheet_name=0, header=0)
print(dataset)
voltype = input("Please enter the type of volatility(weekly/monthly):\n")
if voltype == "weekly":
    div = 5
elif voltype == "monthly":
    div = 20


