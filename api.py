import uvicorn
from fastapi import FastAPI
import pickle
from AdvData import AdvData


# Create the app object
app = FastAPI()

# Load the models
pickle_in = open("NB_model.pickle", "rb")
classifier = pickle.load(pickle_in)

# Sample home get api


@app.get('/')
def index():
    return {'message': 'Heloo!'}

# Api call for prediction

# Sample Request body

# {
#   "Daily_Time_Spent_on_Site": 12,
#   "Age": 33,
#   "Area_Income": 23322,
#   "Daily_Internet_Usage": 23,
#   "Male": 0,
#   "City_Codes": 961,
#   "Country_Codes": 215,
#   "Month": 4,
#   "Day_of_the_month": 4,
#   "Day_of_the_week": 3,
#   "Hour": 3
# }


@app.post('/predict')
def predict(data: AdvData):
    data = data.dict()
    Daily_Time_Spent_on_Site = data['Daily_Time_Spent_on_Site']
    Age = data['Age']
    Area_Income = data['Area_Income']
    Daily_Internet_Usage = data['Daily_Internet_Usage']
    Male = data['Male']
    City_Codes = data['City_Codes']
    Country_Codes = data['Country_Codes']
    Month = data['Month']
    Day_of_the_month = data['Day_of_the_month']
    Day_of_the_week = data['Day_of_the_week']
    Hour = data['Hour']
    prediction = classifier.predict([[Daily_Time_Spent_on_Site, Age, Area_Income, Daily_Internet_Usage,
                                      Male, City_Codes, Country_Codes, Month, Day_of_the_month, Day_of_the_week, Hour]])

    if (prediction[0] == 0):
        prediction = "Ad is not clicked"
    else:
        prediction = "Ad is cliked"

    return {
        'Prediction Results ': prediction,
    }


# Run the API with uvicorn - command : uvicorn api:app --reload
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
