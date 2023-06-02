import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import joblib
import csv
import json
import os

class UserManage():
    def __init__(self):
        self.users = None
        self.load_users()

    def create_user(self, username, password):
        if username in self.users:
            print("Username used")
            return

        self.users[username] = {'password': password, 'logs': []}
        self.save_users()

    def add_log(self, username, in_bed_time, asleep_time, avg_ae, wake_time):
        """
        Add log for user
        : in_bed_time: in hours (float)
        : asleep_time: in hours (float)
        : wake_time: in time format string (HH:MM)
        """

        # reached 30 logs, pop the first log
        if len(self.users[username]['logs']) == 30:
            self.users[username]['logs'].pop(0)
        
        self.users[username]['logs'].append([avg_ae, in_bed_time, asleep_time, wake_time])
        self.save_users()

    def load_users(self):
        try:
            with open('users_log.json', 'r') as f:
                self.users = json.load(f)
        except FileNotFoundError:
            self.users = {}
        
        except ValueError:
            self.users = {}

    def save_users(self):
        with open('users_log.json', 'w') as f:
            json.dump(self.users, f)

    def login(self, username, password):
        if username in self.users:
            if self.users[username]['password'] == password:
                return True
        
        return False

class SleepRecommendation():
    def __init__(self):
        self.sleep_ctg = {(1, 2): (11, 14), 
                          (3, 5): (10, 13), 
                          (6, 13): (9, 11),
                          (14, 17): (8, 10),
                          (18, 25): (7, 9),
                          (26, 64): (7, 9),
                          (65, 150): (7, 8)}
        
        self.user_manage = UserManage()
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def _train_model(self, username):
        logs = self.user_manage.users[username]['logs']

        if len(logs) == 0:
            raise ValueError(f"Not enough data to train the model. Logs contains {len(logs)} values")

        # prepare data for training
        X, y = [], []

        for log in logs:
            avg_ae, in_bed_time, asleep_time, wake_time = log
            wake_time_minutes = self._time_to_minutes(wake_time)
            # check if wake time is next day
            in_bed_time_minutes = in_bed_time * 60
            if wake_time_minutes < in_bed_time_minutes:
                wake_time_minutes += 24 * 60  # add 24 hours

            X.append([avg_ae, in_bed_time_minutes, asleep_time])
            y.append(wake_time_minutes)

        X = self.scaler.fit_transform(X)
        y = np.array(y)

        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)

        self.poly = poly_features
        self.model = LinearRegression()
        self.model.fit(X_poly, y)

        if len(X) > 5:
            # split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # train model
            self.model.fit(X_train, y_train)

            # evaluate model
            score = self.model.score(X_val, y_val)
            print(f"Model Score: {score}")

        else:
            self.model.fit(X, y)

        return True
    

    def _predict_wake_time(self, username, in_bed_time, asleep_time, avg_ae):
        # check if asleep time is larger than in-bed time
        if asleep_time > in_bed_time:
            raise ValueError("Asleep time cannot be larger than in-bed time.")
        
        # train a new model before prediction
        self._train_model(username)
        
        # prepare data for prediction
        in_bed_time_minutes = in_bed_time * 60
        X = self.scaler.transform([[avg_ae, in_bed_time_minutes, asleep_time]])
        wake_time_minutes = self.model.predict(X)[0]
        print(wake_time_minutes)
        if wake_time_minutes < in_bed_time_minutes:
            wake_time_minutes += 24 * 60

        wake_time_minutes %= 24 * 60  # make sure in 24hrs range
        return self._minutes_to_time(wake_time_minutes)
    

    def recommend_sleep_times(self, username, password, in_bed_time, asleep_time, avg_ae, wake_time):
        """
        By calling this method, you will get a list of times for sleep ranked
        by sleep efficiency score as well as the predicted sleep time learned by
        the user's log. Each call of thie method will record the inputs to user's log

        return: (sleep_times_list, predicted_wake_time)
        """
        
        _add_flag = False
        # user access verify
        if self.user_manage.login(username, password) == False:
            raise ValueError("Invalid username or password")
        
        # initial add access data to user's log
        logs = self.user_manage.users[username]['logs']

        if len(logs) == 0:
            self.user_manage.add_log(username, in_bed_time, asleep_time, avg_ae, wake_time)
            _add_flag = True

        # predict wake time for user
        predict_wake_time = self._predict_wake_time(username, in_bed_time, asleep_time, avg_ae)

        # calculate sleep efficiency for each log
        efficiencies = []
        for log in logs:
            log_avg_ae, log_in_bed_time, log_asleep_time, log_wake_time = log
            # calculate sleep efficiency score
            efficiency = log_asleep_time / log_in_bed_time
            wake_time_minutes = self._time_to_minutes(log_wake_time)
            in_bed_time_minutes = log_in_bed_time * 60

            if wake_time_minutes < in_bed_time_minutes:
                wake_time_minutes += 24 * 60

            sleep_time_minutes = wake_time_minutes - in_bed_time_minutes
            efficiencies.append((efficiency, sleep_time_minutes))

        # sort by sleep efficiency score
        efficiencies.sort(reverse=True)

        # get top 3 times
        top_times = efficiencies[:3]

        # convert time format
        rcm_sleep_times = [(self._minutes_to_time(time % (24 * 60)), e) for e, time in top_times]

        # add user acess data to log
        if _add_flag == False:
            self.user_manage.add_log(username, in_bed_time, asleep_time, avg_ae, wake_time)
        
        return rcm_sleep_times, predict_wake_time


    def _time_to_minutes(self, time_str):
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    

    def _minutes_to_time(self, minutes):
        h = minutes // 60
        m = minutes % 60

        return f"{int(h):02}:{int(m):02}"
    
    

if __name__ == "__main__":
    # user_manage = UserManage()
    # user_manage.create_user('dev', 'pass1234')
    # user_manage.create_user('dev2', 'pass2312')
    # user_manage.add_log('dev', 8, 7.5, 120, "7:00")
    # user_manage.add_log('dev', 9, 6.5, 343, "8:00")

    sleep_rcm = SleepRecommendation()
    #sleep_rcm._train_model('dev')
    #sleep_rcm.train_model('dev2')
    print(sleep_rcm.recommend_sleep_times('dev2', 'pass2312', 9, 6.89, 654, "6:32"))

