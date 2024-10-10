# encoding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score
from datetime import datetime



def train():
    file_path = "data/train_data2.csv"
    data_raw = pd.read_csv(file_path, encoding='utf-8')

    x = data_raw[['employment_status','political','marital_status', 'type', 'upgrade', 'gcnt']]
    y = data_raw['petition']
    y=y.astype('int')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    best_params = {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
    xgbc= XGBClassifier(**best_params)
    xgbc = xgbc.fit(x_train, y_train)

    model_file = f"model/petition_model.json"
    xgbc.save_model(model_file)

    y_predict=xgbc.predict(x_test)

    labels = [0, 1] 

    accuracy = accuracy_score(y_test, y_predict, normalize=True)
    recall = recall_score(y_test, y_predict, average='macro', labels=labels)
    f1_value = f1_score(y_test, y_predict, average='macro', labels=labels)

    return accuracy, recall, f1_value

if __name__=="__main__":
    accuracy, recall, f1_value = train()
    print(accuracy, recall, f1_value)

