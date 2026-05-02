import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


df = pd.read_csv("Data.csv")
print(df)

df.columns = df.columns.str.strip()


district_column = "District"
df = df.dropna()


unique_districts = df[district_column].unique()


colors = plt.get_cmap('tab20', len(unique_districts))
markers = ['o', 's', '^', 'D', 'P', '*', 'v', '<', '>', 'H', 'h', '8', 'p', 'x', '_', '+', '1', '2']
linestyles = ['-', '--', '-.', ':']

future_years = np.arange(1991, 2011)


def get_max_population(attribute):
    max_val = 0
    for district in unique_districts:
        pop = df[df[district_column] == district][attribute].values
        if len(pop) > 0:
            max_val = max(max_val, max(pop))
    return max_val

max_total = get_max_population("Total")
max_males = get_max_population("Male")
max_females = get_max_population("Female")
global_ymax = max(max_total, max_males, max_females) * 1.1  



def plot_population(attribute, title, ylabel):
    plt.figure(figsize=(10, 6))
    for idx, district in enumerate(unique_districts):
        district_data = df[df[district_column] == district]
        years = district_data["Year"].values
        pop = district_data[attribute].values
        t = years - years.min()

        
        params, _ = curve_fit(lambda t, P0, r: P0 * np.exp(r * t), t, pop, p0=[pop[0], 0.01])
        P0_fit, r_fit = params
        t_future = future_years - years.min()
        predictions = P0_fit * np.exp(r_fit * t_future)

        
        plt.plot(
            future_years,
            predictions,
            label=district,
            color=colors(idx),
            linestyle=linestyles[idx % len(linestyles)],
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Year", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.ylim(0, global_ymax)  
    plt.legend(title="Districts", loc='upper right', fontsize=8, ncol=2, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



plot_population("Total", "Total Population Growth (1991-2011)", "Total Population")
plot_population("Male", "Male Population Growth (1991-2011)", "Male Population")
plot_population("Female", "Female Population Growth (1991-2011)", "Female Population")


future_years = np.arange(1991, 2031)
plot_population("Total", "Total Population Growth (1991-2031)", "Total Population")
plot_population("Male", "Male Population Growth (1991-2031)", "Male Population")
plot_population("Female", "Female Population Growth (1991-2031)", "Female Population")


attributes = ["Total", "Male", "Female"]
stats = []

for district in unique_districts:
    row = {"District": district}
    district_data = df[df[district_column] == district]
    for attr in attributes:
        row[f"{attr}_Mean"] = district_data[attr].mean()
        row[f"{attr}_Median"] = district_data[attr].median()
        row[f"{attr}_Std"] = district_data[attr].std()
    stats.append(row)

stats_df = pd.DataFrame(stats)
print("\n--- Population Statistics Summary (First 15 Districts) ---")
print(stats_df.head(15).round(2))