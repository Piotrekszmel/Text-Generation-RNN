import os
from sentiment_analysis.evaluate import predict_class

tweets, predicted_y, label = predict_class([
    "I am happy",
    "I am sad :(",
    "Poland is a country"],
    [2,,0,1]
    "datastories.twitter",
    300)
os.system("clear")
for predict_y, label in zip(predicted_y, label):
    print(float(predict_y), ' | ', label)
