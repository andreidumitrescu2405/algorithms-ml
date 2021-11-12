import numpy as np 
from matplotlib import pyplot as plt

# # STOCHASTIC GRADIENT DESCENT

# Generate synthetic data
def generate_data(num_samples=30):
    X = np.array(range(num_samples)) # create the number of points
    random_noise = np.random.uniform(10, 40, size=num_samples) # generate noise from a uniform distribution with mean=10 and std=40
    y = 3.5*X + random_noise # y will be 3.5 * X + the noise 
    return X, y

features, target = generate_data(num_samples=30)

# # Plot
plt.scatter(features, target)
plt.title("Pretul casei in functie de suprafata")
plt.xlabel('Suprafata')
plt.ylabel('Pret casa')
plt.show()

# Initializam paramterii dreptei, panta=t0, bias=t1
t0, t1 = 2, 10

def plot_hypothesis(t0, t1, features, target):
  # Definim ipoteza 
  hypothesis = t0*features + t1 

  # Desenam
  plt.plot(hypothesis, label="Current hypothesis", c='r')
  plt.scatter(features, target)
  plt.title(f"Pretul casei in functie de suprafata.T0 = {t0}. T1={t1}")
  plt.xlabel('Suprafata')
  plt.ylabel('Pret casa')
  plt.legend()
  plt.show()
plot_hypothesis(t0, t1, features, target)

# Functia de cost sau loss MSE
def compute_mse(y_gr, y_pred):
	return np.mean((y_pred - y_gr) ** 2)

# Calculam predictiile ipotezei initiale pentru fiecare intrare x
y_pred = t0*features + t1 

# Calculam eroarea medie patratica intre predictii si target
error = compute_mse(y_pred, target)
print(f"Eroarea patratica medie: {error}")

# Lista in care vom stoca valorile loss-ului din fiecare epoca
errors_per_epoch = []

# Numarul de epoci
epochs = 1000

# Rata de invatare
alpha = 0.0003

# Ne definim perechile de date ca si lista de (feature, target)
datapoints = [(x, y) for x, y  in zip(features, target)]

for _ in range(epochs):
  temp_error = []
  for feature, feature_target in datapoints:
    
    # Predictia
    prediction = t0*feature + t1
    
    # Calculul erorii
    error = compute_mse(feature_target, prediction)
    temp_error.append(error)

    # Gradient descend # 
    # Calcul gradienti folosind formula de mai sus
    t0_grad = feature * (prediction - feature_target)
    t1_grad = (prediction - feature_target)
    
    # Update ponderi
    t0 = t0 - alpha * t0_grad
    t1 = t1 - alpha * t1_grad
  errors_per_epoch.append(np.mean(temp_error))

plot_hypothesis(t0, t1, features, target)

plt.xlabel("Epoca")
plt.ylabel("Eroare medie")
plt.title("Functia de loss")
plt.plot(errors_per_epoch)
plt.show()
print(f"Eroarea finala: {np.mean(temp_error)}")