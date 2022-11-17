def uniformly_choose_hyp(manifold, X ,delta):
    return delta * manifold.random_tangent_vector(X)