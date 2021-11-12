import numpy as np 
from matplotlib import pyplot as plt


def compute_mse(y_gr, y_pred):
	return np.mean((y_pred - y_gr) ** 2)
# Generate data that is not liniar
def generate_nonliniar_data():
  features = np.random.rand(1000) 
  target = 8*((features)**(2)) + np.random.rand(1000)
  return features, target

features, target = generate_nonliniar_data()

plt.scatter(features, target)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# Lista in care vom stoca valorile loss-ului din fiecare epoca
errors_per_epoch = []

# Numarul de epoci
epochs = 1000

# Rata de invatare 
alpha = 0.003

# Ne definim perechile de date ca si lista de (feature, target)
datapoints = [(x, y) for x, y  in zip(features, target)]

# Initializam ponderile
t0, t1, t2 = 1.5, 2, 1.5
# t3 = 1.7

for _ in range(epochs):
  temp_error = []
  for feature, feature_target in datapoints:
    
    # Predictia
    prediction = t0*feature**2 + t1 * feature + t2  
    
    # Calculul erorii
    error = compute_mse(feature_target, prediction)
    temp_error.append(error)

    # Gradient descend ->  ti = ti - learning_rate*gradient_ti
    t0 = t0 - alpha * (feature**2 * (prediction - feature_target))
    t1 = t1 - alpha * (feature * (prediction - feature_target))
    t2 = t2 - alpha * (prediction - feature_target)
    # t3 = t3 - alpha * (prediction - feature_target)

  errors_per_epoch.append(np.mean(temp_error))

# Plot
plt.xlabel("Epoca")
plt.ylabel("Eroare medie")
plt.title("Functia de loss")
plt.plot(errors_per_epoch)
plt.show()

# Plot-uim din nou functia obtinuta in raport cu datele de antrenare
preds = t0*features**2 + t1 * features + t2 

plt.rcParams["figure.figsize"] = (8, 8)
plt.xlabel("Epoca")
plt.ylabel("Eroare medie")
plt.title("Functia de loss")
plt.scatter(features, target, zorder=2)
plt.scatter(features, preds, zorder=10, s=2.5, c='r')
plt.show()
print(f"Eroarea medie ultima epoca: {compute_mse(preds, target)}")