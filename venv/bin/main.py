import pandas as pd
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import numpy as np

# Import data
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)


#Take a peek at the data
print(df.head())

# Replace ? to nan
df.replace("?", np.nan, inplace = True)


# # Check our missing data and see what is missing
# missingData = df.isnull()
#
# for column in missingData.columns.values.tolist():
#     print(column)
#     print(missingData[column].value_counts())
#     print("")

# There is missing data in normalized-losses, num-of-doors, bore, stroke, horsepower, peak-rpm, and price


# Question One: Replace NaN in the "stroke" column to the mean of the col
mean = df["stroke"].astype(float).mean(axis=0)

df["stroke"].replace(np.nan, mean, inplace=True)


# Handle missing normalized-losses. Mean method makes sense here since it is a continuous variable

mean = df["normalized-losses"].astype(float).mean(axis=0)

df["normalized-losses"].replace(np.nan, mean, inplace=True)

# Handle missing num-of-doors. Makes sense to use mode here since categorical variable

mode = df["num-of-doors"].value_counts().idxmax()

df["num-of-doors"].replace(np.nan, mode, inplace=True)

# Handle missing bore. Mean method

mean = df["bore"].astype(float).mean(axis=0)

df["bore"].replace(np.nan, mean, inplace=True)

# Handle missing horsepower. Mean method

mean = df["horsepower"].astype(float).mean()

df["horsepower"].replace(np.nan, mean, inplace=True)

# Handle missing peak-rpm. Mean method

mean = df["peak-rpm"].astype(float).mean()

df["peak-rpm"].replace(np.nan, mean, inplace=True)

# Handle missing price. Since this is the Label, we should just drop the rows rather than guess

df.dropna(axis=0, subset=["price"], inplace=True)

df.reset_index(drop=True, inplace=True) # resets the indicies after the drop

# # Should have no missing data now
#
# finalMissingData = df.isnull()
# for column in finalMissingData.columns.values.tolist():
#     print(column)
#     print(finalMissingData[column].value_counts())
#     print("")

# We should fix the types now

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df[["horsepower"]] = df[["horsepower"]].astype("int")

# Data is now cleaned
# Time for standardization (metric units)

df["city-L/100km"] = 235 / df["city-mpg"]

df["highway-L/100km"] = 235 / df["highway-mpg"]

#Normalize using simple feature scaling
df["length"] = df["length"] / df["length"].max()
df["width"] = df["width"] / df["width"].max()
df["height"] = df["height"] / df["height"].max()



#Binning. Lets make a plot of horsepower

pyplot.hist(df["horsepower"])

pyplot.xlabel("Horsepower")
pyplot.ylabel("count")
pyplot.title("Horsepower Distribution")

pyplot.show()



# We want 5 bins (not a great split but just for an example it should suffice) instead of 57
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 6)

groupNames = ["low", "low-med", "medium", "med-high", "high"]

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=groupNames, include_lowest=True )

pyplot.bar(groupNames, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
pyplot.xlabel("horsepower")
pyplot.ylabel("count")
pyplot.title("horsepower bins")

pyplot.show()




# Indicator variable for categortical variable, fuel-type
dummyOne = pd.get_dummies(df["fuel-type"])

# bring them together
df = pd.concat([df, dummyOne], axis=1)
df.drop("fuel-type", axis=1, inplace=True)



# Create dummy for the aspirator too
dummyTwo = pd.get_dummies(df["aspiration"])

df = pd.concat([df,dummyTwo], axis=1)
df.drop("aspiration", axis=1, inplace=True)

#Export cleaned data

df.to_csv("clean_data.csv")




