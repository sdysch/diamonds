import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#====================================================================================================

def main():

    DATA_FILE = "data/diamonds.csv"

    # read data
    data = pd.read_csv(DATA_FILE)

    # plot some distributions

    # carat
    fig_carat, ax_carat = plt.subplots()
    hist_carat = ax_carat.hist(data["carat"], bins = 20)
    ax_carat.set_xlabel("carat")
    ax_carat.set_ylabel("Events/bin")
    fig_carat.savefig("carat.png")


    # x, y, z lengths
    lengths, axs = plt.subplots(3, 1, figsize = (7, 10))
    axs[0].hist(data["x"], bins = 15, color = "blue")
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("Events/bin")

    axs[1].hist(data["y"], bins = 50, color = "blue")
    axs[1].set_xlabel("y [mm]")
    axs[1].set_ylabel("Events/bin")

    axs[2].hist(data["z"], bins = 50, color = "blue")
    axs[2].set_xlabel("z [mm]")
    axs[2].set_ylabel("Events/bin")

    lengths.savefig("lengths.png")

    # depth (z / mean(x, y))
    fig_depth, ax_depth = plt.subplots()
    hist_depth = ax_depth.hist(data["depth"], bins = 20)
    ax_depth.set_xlabel("depth")
    ax_depth.set_ylabel("Events/bin")
    fig_depth.savefig("depth.png")

    # some preprocessing
    # diamond clarity runs from I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
    # replace with ascending integers
    #enc = LabelEncoder()
    clarity_dictionary_replace = {
        "I1"   : 0,
        "SI2"  : 1,
        "SI1"  : 2,
        "VS2"  : 3,
        "VS1"  : 4,
        "VVS2" : 5,
        "VVS1" : 6,
        "IF"   : 7
    }

    data["clarity"].replace(clarity_dictionary_replace, inplace = True)

    # diamond cut (in preferential order) Fair, Good, Very Good, Premium, Ideal
    cut_dictionary_replace = {
        "Fair"      : 0,
        "Good"      : 1,
        "Very Good" : 2,
        "Premium"   : 3,
        "Ideal"     : 4
    }
    data["cut"].replace(cut_dictionary_replace, inplace = True)

    # label encoding for diamond colour
    enc = LabelEncoder()
    data["color"] = enc.fit_transform(data["color"])

    # drop index row
    data = data.drop(["Unnamed: 0"], axis = 1)

    # drop x,y,z in favour of depth? check!
    data = data.drop(["x", "y", "z"], axis = 1)

    # split data, predict price
    X = data.drop(["price"], axis = 1)
    y = data["price"]
    #print(X)
    #print(y)

    # split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # regression methods

    # linear regression
    regr = LinearRegression()
    model = regr.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("root mean squared error: {err}".format(err = np.sqrt(mean_squared_error(y_test, y_pred))))
    print("r-squared: {r2} ".format(r2 = np.sqrt(r2_score(y_test, y_pred))))

    # k-neighbours
    kneighbours = KNeighborsRegressor(n_neighbors = 4, weights = "distance")
    model = kneighbours.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("root mean squared error: {err}".format(err = np.sqrt(mean_squared_error(y_test, y_pred))))
    print("r-squared: {r2} ".format(r2 = np.sqrt(r2_score(y_test, y_pred))))


#====================================================================================================

if __name__ == "__main__":
    main()
