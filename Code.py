import pandas as pd
import numpy as np
# loading dataset
dataset = pd.read_csv(r"D:\college\ItemRecommendation\shopping_trends.csv")
print("Dataset loaded!")
print(dataset["Season"].unique())

#DROPPING UNNECESSARY COLUMNS

dataset.drop(["Subscription Status","Shipping Type","Payment Method","Customer ID"],inplace=True,axis=1)

# SEGREGATING IN AGE GROUPS
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-100"]
dataset["Age Group"] = pd.cut(dataset["Age"], bins=bins, labels=labels)
print("Customers segregated into age groups successfully.")

# ENCODING CATEGORICAL COLUMNS 
from sklearn.preprocessing import LabelEncoder

le_gender=LabelEncoder()
le_gender.fit(dataset["Gender"])
le_item_purchased=LabelEncoder()
le_item_purchased.fit(dataset["Item Purchased"])
le_size=LabelEncoder()
le_size.fit(dataset["Size"])
le_color=LabelEncoder()
le_color.fit(dataset["Color"])
le_season=LabelEncoder()
le_season.fit(dataset["Season"])
le_category=LabelEncoder()
le_category.fit(dataset["Category"])
le_location=LabelEncoder()
le_location.fit(dataset["Location"])
le_discount=LabelEncoder()
le_discount.fit(dataset["Discount Applied"])
le_purchase_frequency=LabelEncoder()
le_purchase_frequency.fit(dataset["Frequency of Purchases"])
le_age_group=LabelEncoder()
le_age_group.fit(dataset["Age Group"])
dataset["Gender"] = le_gender.transform(dataset["Gender"])
dataset["Item Purchased"] = le_item_purchased.transform(dataset["Item Purchased"])
dataset["Size"] = le_size.transform(dataset["Size"])
dataset["Color"] = le_color.transform(dataset["Color"])
dataset["Season"] = le_season.transform(dataset["Season"])
dataset["Category"] = le_category.transform(dataset["Category"])
dataset["Location"] = le_location.transform(dataset["Location"])
dataset["Discount Applied"] = le_discount.transform(dataset["Discount Applied"])
dataset["Frequency of Purchases"] = le_purchase_frequency.transform(dataset["Frequency of Purchases"])
dataset["Age Group"] = le_age_group.transform(dataset["Age Group"])
print("Categorical columns encoded successfully.")

# DEFINING FEATURES AND LABELS
x = dataset[["Age Group", "Location", "Season", "Review Rating", "Gender"]]
y = dataset["Item Purchased"]

# CREATING AN UPDATED CSV FILE
updated_csv = r"D:\college\ItemRecommendation\updated_file.csv"
dataset.to_csv(updated_csv, index=False)

# APPLYING KNN MODEL
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(x)
print("Applied K nearest neighbor model successfully!")


#TAKING USER INPUTS
age=int(input("Enter your age : "))

if age <= 20:
    age_group = "0-20"
elif age <= 30:
    age_group = "20-30"
elif age <= 40:
    age_group = "30-40"
elif age <= 50:
    age_group = "40-50"
elif age <= 60:
    age_group = "50-60"
else:
    age_group = "60-100"

gender=input("Enter your gender(Male/Female) : ")

import datetime
now=datetime.datetime.now()
month=now.month
if month in [3,4,5]:
    season="Spring"
elif month in [12,1,2]:
    season="Winter"
elif month in [6,7,8]:
    season="Summer"
else:
    season="Fall"

location=input("Enter your location : ")
try:
    age_group=le_age_group.transform([age_group])[0]
    season=le_season.transform([season])[0]
    gender=le_gender.transform([gender])[0]
    location=le_location.transform([location])[0]
except ValueError as e:
    print(f"Error in input encoding: {e}")
    exit(1)
    
#RECOMMENDING
user_input=np.array([[age_group,location,season,5,gender]])
distances, indices=model.kneighbors(user_input)
print("Recommendations based on your profile : ")

for i in indices[0]:
    item=y.iloc[i]
    item_name = le_item_purchased.inverse_transform([item])[0]
    print(item_name)
