import numpy as np
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            #print(idx, t, corpus[idx + t])
            cs.append(corpus[idx + t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)

#corpus = np.array([0, 1, 2, 3, 4, 5, 6])

#contexts, target = create_contexts_target(corpus, window_size=1)
#print(contexts)