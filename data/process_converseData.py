import pandas as pd

labels = {"중립":0, "행복":1, "슬픔":2, "놀람":3, "분노":4, "공포":5, "혐오":6}

# Read ConverseData excel file
train_file = "ConverseData.xlsx"
data = pd.read_excel(train_file)
queries = data['Sentence'].tolist()
intents = data['Emotion'].tolist()
intents_ = []

# Convert the Korean intent(Emotion) labels into integer labels
for intent in intents:
	intents_.append(labels[intent])

# Save the converted data into csv file
processed_data = {'query': queries, 'intent': intents_}
df = pd.DataFrame(processed_data)
df.to_csv('ConverseData.csv')