import csv

with open('sport.csv') as f:
    csv_file = csv.reader(f)
    data = list(csv_file)
    print(data)
    specific = data[1][:-1]
    general = [['?' for i in range(len(specific))] for j in range(len(specific))]


for i in data:
    if i[-1] == "Yes":
      for j in range(len(specific)):
        if i[j] != specific[j]:
          specific[j] = "?"
          general[j][j] = "?"

    elif i[-1] == "No":
      for j in range(len(specific)):
        if i[j] != specific[j]:
          general[j][j] = specific[j]
        else:
          general[j][j] = "?"

    print("\nStep " + str(data.index(i)+1) + " of Candidate Elimination Algorithm")
    print(specific)
    print(general)


gh = [] #General Hypothesis
for i in general:
  for j in i:
    if j != '?':
      gh.append(i)
      break
print("\nFinal Specific Hypothesis: ", specific)
print("\nFinal General Hypothesis: ", gh)