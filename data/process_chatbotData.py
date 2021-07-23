import pandas as pd

# Read ChatbotData file
train_file = "ChatbotData.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['Q'].tolist()
intents = data['label'].tolist()

# Save the converted data into csv file
processed_data = {'query': queries, 'intent': intents}
df = pd.DataFrame(processed_data)
df.to_csv('ChatbotData_.csv')