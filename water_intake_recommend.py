import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import csv
import os

class WaterRecommendatio():
    def __init__(self, input_path):
        """
        : input_path -> path to dataset
        """
        self._init_prediction_model(input_path, file_name='water_intake_model', out_path='models')
        self.input_path = input_path

    def _init_water_df(self, filepath):
        """
        This method extract required data from csv,
        call this if the dataframe is not prepared
        """
        return pd.read_csv(filepath)

    
    def _init_prediction_model(self, input_path, file_name, out_path, debug=False):
        """
        This menthod pre-trained the regression model for prediction
        the model will be saved if not exist
        : input_path -> path to dataset
        : file_name -> name of the output model
        : out_path -> path folder to save model
        """
        model_folder = out_path
        model_file = file_name
        model_path = os.path.join(model_folder, model_file)

        # save model if not exists
        if not os.path.exists(model_path):
            self._water_df = self._init_water_df(input_path)
            intake_X = self._water_df.iloc[:, :-1]  # features of intake data
            intake_y = self._water_df['water_intake']  # outputs of intake data

            # scale data
            scaler = StandardScaler()
            intake_X_scaled = scaler.fit_transform(intake_X)

            # create 80%/20% train/test split
            intake_X_tr, intake_X_te, intake_y_tr, intake_y_te = train_test_split(intake_X_scaled, intake_y, test_size=0.2, random_state=0)

            # find best k value
            knn = KNeighborsRegressor()
            param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
            grid_search = GridSearchCV(knn, param_grid, cv=3)
            grid_search.fit(intake_X_tr, intake_y_tr)
            best_k = grid_search.best_params_['n_neighbors']

            # initialize KNN reg model
            model = KNeighborsRegressor(n_neighbors=best_k)
            model.fit(intake_X_tr, intake_y_tr)

            if debug:
                print("The best k value is: {}".format(best_k))
                y_tr_pred = model.predict(intake_X_tr)
                mse = mean_squared_error(intake_y_tr, y_tr_pred)
                print(f"training mse is {mse}")

                y_te_pred = model.predict(intake_X_te)
                mse = mean_squared_error(intake_y_te, y_te_pred)
                print(f"testing mse is {mse}")

            # save the model
            self._save_model(model, file_name, out_path)
            print("model saved")

            # save standard scaler
            scaler_path = os.path.join(model_folder, 'std_scaler.bin')
            joblib.dump(scaler, scaler_path, compress=True)
            print("scaler saved")
         
    def _save_model(self, model, file_name, out_path):
        """
        Save the model to path if the model is not the same.
        The number in the end of filename indicates the number of datapoints that we have in training file
        if the new model file has different value, it means the model is not the same as preivous one
        """
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        file_path = os.path.join(out_path, file_name)
        joblib.dump(model, file_path)

    
    def recommend_water_intake(self, age, weight, height, avg_ae, temperature, model_path, scaler_path):
        """
        : model_path -> path to model
        : scaler_path -> path to scaler
        """
        activity_level = None
        # determine activity level (sedentary, lightly active, active, very active)
        if (avg_ae < 130):
            activity_level = 1.2
        elif (avg_ae >= 130 and avg_ae < 160):
            activity_level = 1.375
        elif (avg_ae >= 160 and avg_ae < 580):
            activity_level = 1.55
        else:
            activity_level = 1.725


        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_df = pd.DataFrame({"age": [age], "weight": [weight], "height": [height], "activity_level": [activity_level], "temperature": [temperature]})
        X_scaled = scaler.transform(X_df)
        intake_level = model.predict(X_scaled)

        return float(intake_level)


    def _save_actual_intake(self, age, weight, height, avg_ae, temperature, actual_intake):
        with open(self.input_path, 'w', newline='') as file:
            file.writable


if __name__ == "__main__":
    path = 'water_intake_datasets/water_drinking_data.csv'
    wr = WaterRecommendatio(input_path=path)

    model_path = 'models/water_intake_model'
    scaler_path = 'models/std_scaler.bin'
    result = wr.recommend_water_intake(21, 10, 1.85, 320, 14, model_path, scaler_path)
    print(f"Should drink {result} liters of water")



