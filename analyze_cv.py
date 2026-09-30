import json
from collections import defaultdict

with open('c:\\Users\\Ronaldo\\Notas\\Art_Sant\\mining-digital-twin\\backend\\data\\artifacts\\cv_results.json') as f:
    data = json.load(f)

results = data['results']
acc = defaultdict(list)
for r in results:
    acc[r['Arquitectura']].append(float(r['Accuracy']))

print("Accuracy promedio por modelo:")
for arch, values in sorted(acc.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
    print(f"{arch}: {sum(values)/len(values):.4f}")

best_arch = max(acc.items(), key=lambda x: sum(x[1])/len(x[1]))
print(f"\nMejor modelo: {best_arch[0]} con accuracy {sum(best_arch[1])/len(best_arch[1]):.4f}")
