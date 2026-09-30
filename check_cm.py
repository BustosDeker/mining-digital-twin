import json
import numpy as np
from sklearn.metrics import confusion_matrix

with open('c:\\Users\\Ronaldo\\Notas\\Art_Sant\\mining-digital-twin\\backend\\data\\artifacts\\cv_results.json') as f:
    data = json.load(f)

results = data['results']

# Check features_mlp
model_results = [r for r in results if r["Arquitectura"] == "features_mlp"]
if model_results:
    class_names = model_results[0]["class_names"]
    n_classes = len(class_names)
    cm_agg = np.zeros((n_classes, n_classes), dtype=int)
    
    for r in model_results:
        y_true = np.array(r["y_true"])
        y_pred = np.array(r["y_pred"])
        cm_agg += confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    print(f"Matriz de confusión agregada para features_mlp:")
    print(cm_agg)
    print(f"\nTotal predicciones: {cm_agg.sum()}")
    print(f"Clases: {class_names}")
