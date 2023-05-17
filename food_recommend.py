import json
import sys
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skewnorm

class FoodRecommendation():
    def __init__(self, filepath):
        """
        : filepath: path to json
        """
        self._food_df = None

        self._init_food_df(filepath)
    

    def _init_food_df(self, filepath):
        """
        This method extract required data from JSON,
        call this if the dataframe is not prepared
        : filepath -> path to json file
        """

        with open(filepath, 'r') as file:
            data = json.load(file)


        food_list = []
        # TODO: this may need to modify to accommodate other json formats
        for entry in data['all']:
            for menu_item in entry['menu']:
                for food_item in menu_item['items']:
                    if 'calories' in food_item['nutrition'] and food_item['nutrition']['calories'] != None:
                        food = {
                            'type': menu_item['category'],
                            'name': food_item['name'],
                            'calories': float(food_item['nutrition']['calories']),
                        }
                        food_list.append(food)
        
        self._food_df = pd.DataFrame(food_list)
        # convert food type into categorical values
        le = LabelEncoder()
        self._food_df['type'] = le.fit_transform(self._food_df['type'])
        

    def _calc_calneeds(self, weight, fat, avg_ae, time='lunch'):
        """
        This method calculate and init the calories need for the user.
        Recommend Cal Needs = BMR + Average Activity Level
        : fat -> body fat percentage
        : avg_ae -> average active energy
        : time -> breakfast, lunch, or dinner
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



        cal_splits = {'breakfast': 0.2, 'lunch': 0.4, 'dinner': 0.4, 'all': 1}
        bmr = 370 + 21.6 * (1-fat) * weight
        cal_needs = bmr * activity_level
        cal_needs = cal_needs * cal_splits[time]

        return cal_needs
    

    def find_foods(self, input_cal, c=1, debug=False):
        """
        Find k top food that have closest calories to the user calories
        : c - size of combinations of food items
        """

        # random split of input calories to get different combinations of food
        def _skewed_random(a, b, skewness=2, size=None):
            # usually one food of a meal should have larger weight (e.g. primary dish)
            # therefore we need to skew the splits
            loc = (a + b) / 2
            scale = (b - a) / 6
            a_param = skewness
            x = skewnorm.rvs(a_param, loc, scale, size)
            return x
        
        # generate c random calories splits (skewed)
        cal_seg = []
        total_cal = input_cal
        for _ in range(c):
            cal = _skewed_random(0, total_cal, skewness=-2)
            cal_seg.append(cal)
            total_cal -= cal

        if debug == True:
            print(cal_seg)

        # find c foods closest to the c splits
        cdist = [(0, None)] * c
        for idx, cal in enumerate(cal_seg):
            min_cal = sys.float_info.max
            for _, row in self._food_df[['name', 'calories']].iterrows():
                dist = np.abs(cal - row.values[1])
                
                if dist < min_cal:
                    cdist[idx] = (dist, row.values[0])
                    min_cal = dist

        return cdist
    

    def recommend_foods(self, weight=70, fat=0.165, avg_ae=300, time='lunch', c=1):
        """
        Recommend c foods based on user's health data
        : c - size of combinations of food items
        """
        cal_needs = self._calc_calneeds(weight, fat, avg_ae, time)

        return self.find_foods(cal_needs, c)
    

if __name__ == "__main__":
    path = 'food_datasets/test.json'
    fr = FoodRecommendation(path)
    foods = fr.recommend_foods(weight=55, fat=0.165, avg_ae=0, time='lunch', c=3)

    for dist, food_name in foods:
        print(f"\033[1mFood Name\033[0m: {food_name}")




