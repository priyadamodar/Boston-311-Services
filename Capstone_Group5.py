"""
ALY6140 Analytics Systems Technology Fall B 2020
College of Professional Studies, Northeastern University, Bpston
WEEK 6 : Capstone Project Group 5
Submission date : 12/11/20
Copyright (c) 2020
Licensed
Written by <Devipriya Damodar, Suvarna kumar Doodala, Krishna Chaitanya Rokkam>
Instructor name : Richard He

"""
import os
os.getcwd()
os.chdir("/Users/priya/Downloads/data")
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""This fuction is created to Clean th data"""

def clean():
    
   df['neighborhood'] = df['neighborhood'].fillna("No Data")
   df['location_zipcode'] = df['location_zipcode'].fillna("00000").astype(int)

 def drop_columns():
     df.drop('submittedphoto',
  axis='columns', inplace=True)
   
def pie_chart():
    df['reason'].value_counts()[:10].sort_values().plot(title='Reason for requests', kind='pie')


def bar_graph():
    df['Open_Month'].value_counts()[:10].sort_values().plot(title='Number of requests per month', kind='bar')


def make_lower_case(s):
    if isinstance(s, str):
        return s.lower()
    return str(s).lower()

def Lable_encoder():
  case_titlelabel_encoder = preprocessing.LabelEncoder()
  reasonlabel_encoder = preprocessing.LabelEncoder()
  subjectlabel_encoder = preprocessing.LabelEncoder()

def divide_the_data()

final_df = pd.DataFrame()
final_df["case_title"] = case_titlelabel_encoder.fit_transform(df.case_title.apply(make_lower))
final_df["reason"] = reasonlabel_encoder.fit_transform(df.reason.apply(make_lower))
final_df["subject"] = subjectlabel_encoder.fit_transform(df.subject)
X = final_df[final_df.columns[:-1]]
y = final_df[final_df.columns[-1]]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

def Logistic_regression():
    logistic_regression= LogisticRegression(max_iter=500)
    logistic_regression.fit(X_train,y_train)
    y_pred=logistic_regression.predict(X_test)
    print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
    plt.show()
    
 def Random_forest():
     rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
     rf.fit(X_train, y_train)
     predictions = rf.predict(X_test)
     predictions = predictions.astype(int)
     print('Accuracy: ',metrics.accuracy_score(y_test, predictions))
     plt.show()
     
 def KNN():    
     Kmod = KNeighborsClassifier(n_neighbors = 3000)
     Kmod.fit (X_train, y_train)
     prediction = Kmod.predict(X_test)
     print('Accuracy: ',metrics.accuracy_score(y_test, prediction))

 def Test_case():
     Case_title="Requests for Street Cleaning"
    Reason= "Street Cleaning"
    test_case_title = case_titlelabel_encoder.transform(["Requests for Street Cleaning".lower()])[0]
    test_reason = reasonlabel_encoder.transform(["Street Cleaning".lower()])[0]
    testxdf = pd.DataFrame([[test_case_title,test_reason]])

    print("The case title is",Case_title)
    print("The reason mentioned is",Reason)
    print("The output of our models is")

    print("Logistic regression output :  ", end="")
    print(subjectlabel_encoder.inverse_transform(logistic_regression.predict(testxdf)))

    print("Random forest classifier output :  ", end="")
    print(subjectlabel_encoder.inverse_transform(rf.predict(testxdf).astype(int)))

    print("K-Nearest Neighbors output :  ", end="")
    print(subjectlabel_encoder.inverse_transform(Kmod.predict(testxdf).astype(int))) 

# In[3]:
if __name__ == "__main__":
    # Main functions to Run
    ds1 = clean();
    

if __name__ == "__main__":
    # Main functions to Run
    ds2 = drop_columns();
    

if __name__ == "__main__":
    # Main functions to Run
    ds2 = bar_graph();
    
if __name__ == "__main__":
    # Main functions to Run
    ds2 = pie_chart();
    
if __name__ == "__main__":
    # Main functions to Run
    ds2 = make_lower();
    
if __name__ == "__main__":
    # Main functions to Run
    ds2 = Lable_encoder();
    
if __name__ == "__main__":
    # Main functions to Run
    ds2 = drop_columns();
    
if __name__ == "__main__":
    # Main functions to Run
    ds3 = divide_the_data();
   

if __name__ == "__main__":
    # Main functions to Run
    ds4 = Logistic_regression();
    
    
if __name__ == "__main__":
    # Main functions to Run
    ds5 = Random_forest();
    
 if __name__ == "__main__":
    # Main functions to Run
    ds5 = KNN();
    
 if __name__ == "__main__":
    # Main functions to Run
    ds5 = Test_case();
    

