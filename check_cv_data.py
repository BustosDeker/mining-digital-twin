import json

with open('c:\\Users\\Ronaldo\\Notas\\Art_Sant\\mining-digital-twin\\backend\\data\\artifacts\\cv_results.json') as f:
    data = json.load(f)

results = data['results']
first_result = results[0]

print("Primer resultado:")
print(f"Arquitectura: {first_result['Arquitectura']}")
print(f"Fold: {first_result['Fold']}")
print(f"y_true tiene datos: {'y_true' in first_result}")
print(f"y_pred tiene datos: {'y_pred' in first_result}")
print(f"class_names tiene datos: {'class_names' in first_result}")

if 'y_true' in first_result:
    print(f"y_true length: {len(first_result['y_true'])}")
    print(f"y_pred length: {len(first_result['y_pred'])}")
    print(f"class_names: {first_result['class_names']}")
