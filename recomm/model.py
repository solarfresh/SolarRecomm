def factorization_machine(x, w, v):
    """
    Besides a linear (order-1) interactions
    among features, FM models pairwise (order-2) feature interactions
    as inner product of respective feature latent vectors

    :param x: a d-dimensional vector and its elements have m-fields involves
              pairs of users and items
    :param w: a d-dimensional vector which is the weight in the model
    :param v: a d-dimensional vector and its elements are k-dimensional vectors
              which are latency of features
    :return: m-dimensional vector
    """