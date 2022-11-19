from pydantic import BaseModel


class AdvData(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: float
    Area_Income: float
    Daily_Internet_Usage: float
    Male: float
    City_Codes: float
    Country_Codes: float
    Month: float
    Day_of_the_month: float
    Day_of_the_week: float
    Hour: float
