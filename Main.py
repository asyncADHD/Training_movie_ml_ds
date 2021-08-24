import numpy as np


x_train = np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,1,0]])

y_train = ['Y', 'N', 'Y', 'Y']

x_test = np.array([[1, 1, 0]])

def get_label_indices(labels):
    # group sample bassed on their labels and return indices 
    # param lables: list of lables 
    # Return: dict,{class1: [indices], calss2 [incdices]}

    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

label_indices = get_label_indices(y_train)
print ('label_indices:\n', label_indices)

def get_prior(label_indices):
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

prior = get_prior(label_indices)
print ('prior: \n', prior)

def get_likelihood(features, label_indices, smoothing=0):
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

smoothing = 1
likelihood = get_likelihood(x_train, label_indices, smoothing)
print ('likelihood: \n', likelihood)

def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        # posterior is prorportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
        return posteriors


posterior = get_posterior(x_test, prior, likelihood)
print ('poaterior: \n', posterior)


from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=1.0, fit_prior=True)

clf.fit(x_train, y_train)

pred_prob = clf.predict_proba(x_test)
print ('[scikit-learn] Predicted probabilites: \n', pred_prob)

pred = clf.predict(x_test)
print('[scikit-learn] Prdeiction:', pred)
