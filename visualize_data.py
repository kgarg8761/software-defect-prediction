import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("kc1_cleaned.csv")

plt.figure(figsize=(6, 4))
sns.countplot(x="defects", data=data)

plt.title("Defect Distribution")
plt.xlabel("Defect (0 = No Defect, 1 = Defect)")
plt.ylabel("Number of Software Modules")

plt.savefig("defect_distribution.png")
plt.show()
plt.close()


plt.figure(figsize=(7, 5))
sns.boxplot(x="defects", y="loc", data=data)

plt.title("Lines of Code vs Defects")
plt.xlabel("Defect (0 = No Defect, 1 = Defect)")
plt.ylabel("Lines of Code")

plt.savefig("loc_vs_defects.png")
plt.show()
plt.close()


plt.figure(figsize=(7, 5))
sns.boxplot(x="defects", y="v(g)", data=data)

plt.title("Cyclomatic Complexity vs Defects")
plt.xlabel("Defect (0 = No Defect, 1 = Defect)")
plt.ylabel("Cyclomatic Complexity")

plt.savefig("complexity_vs_defects.png")
plt.show()
plt.close()


plt.figure(figsize=(12, 8))

correlation = data.corr()

sns.heatmap(correlation, annot=False, cmap="coolwarm")

plt.title("Correlation Heatmap of KC1 Dataset")

plt.savefig("correlation_heatmap.png")
plt.show()
plt.close()

print("All graphs saved successfully!")
