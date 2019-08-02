import pickle
import csv


with open('result.pkl', 'rb') as fd:
    data = pickle.load(fd)


with open('data.csv', 'w') as fd:
    csv_writer = csv.writer(fd)

    # write header
    csv_writer.writerow(['TagName', 'ExampleCount', 'CorrectRate'])

    for name, tag in data['tags'].items():
        csv_writer.writerow([name, tag['number'], tag['correct_rate']])
