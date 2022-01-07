import numpy as np
from utils import *

# L_k_list: the sets of a to be selected from
# G_list: corresponding aggregated G values
# age_list: corresponding ages
# n: number of sets to be selected

# (1) what else to pass? grad_phi_list, L_k_list, (age)
# (2) long-term history of age across L
# (bonus) long-term history of age across different L_k's

# (bonus) multiple directions to step into & select best/avg
# (bonus) EA on a_0 with "SGD" - step in random directions in a-space, and assess best directions (via gradient) and evolve

def EA_selection(L_k_list, G_list, age_list, a_list, error_list, n):
    fitnesses = dict()
    mapping = dict()
    count = 0
    for L_k in L_k_list:
        fitness = EA_fitness(G_list[count], age_list[count], error_list[count])
        fitnesses[count] = fitness  # [L_k,age_k,a_k] combination maps to a fitness
        mapping[count] = [list(np.sort(L_k)), age_list[count], a_list[count]]
        count += 1
    fitnesses_sorted = dict(sorted(fitnesses.items(), key=lambda item: item[1]))

    L_k_list_to_return = []
    age_list_to_return = []
    a_list_to_return = []
    count = 0
    for index in fitnesses_sorted.keys():
        count += 1
        if not mapping[index][0] in L_k_list_to_return:
            L_k_list_to_return.append(mapping[index][0])
            age_list_to_return.append(mapping[index][1] + 1)
            a_list_to_return.append(mapping[index][2])
        if len(L_k_list_to_return) > n - 1:
            break
    return [L_k_list_to_return, age_list_to_return, a_list_to_return]


def EA_breed(parent_vals, age_vals, a_vals, L_max, split, num_children):
    parent_count = len(parent_vals)
    n = 0.1  # n is the number of parameters to mutate - 10% of points?
    children = []
    updated_population = parent_vals.copy()
    count = 0
    for i in range(parent_count):
        parent_vals[i] = list(parent_vals[i])
    for child in range(num_children):
        baby = parent_vals[0]  # not elegant - is needed to enter loop for first time
        while list(np.sort(baby)) in parent_vals:
            parent1_index = np.random.randint(parent_count)
            parent1 = parent_vals[parent1_index]
            decider = np.random.randn()
            # randomly choose between recombination+mutation or just mutation (weigh towards recombination+mutation)
            if decider > 0.2:
                # i.e. if recombination + mutation
                parent2 = parent_vals[np.random.randint(parent_count)]
                while list(parent1) == list(parent2):
                    # in case self-pollination
                    parent2 = parent_vals[np.random.randint(parent_count)]
                baby = recombine(parent1, parent2, split)
            else:
                # i.e. if only mutation
                baby = parent1
            baby = mutate(baby, L_max)
        count += 1
        updated_population.append(baby)
        age_vals.append(0)
        a_vals.append(a_vals[parent1_index])

    return [updated_population, age_vals, a_vals]


def recombine(parent1, parent2, split):
    # 70% from parent1 & 30% from parent2? can tweak as hyperparameter
    # need to see if lists or numpy arrays work better here
    baby = parent1.copy()
    intersection = set(parent1) & set(parent2)

    parent1_only = list(set(parent1) - intersection)
    parent2_only = list(set(parent2) - intersection)
    intersection = list(intersection)

    # SHOULD SHUFFLE LISTS HERE
    np.random.shuffle(parent1_only)
    np.random.shuffle(parent2_only)
    np.random.shuffle(intersection)

    K = len(parent1)
    intersection_count = 0
    parent2_count = 0
    for i in range(K):
        if i < int(split * K) - 1:
            # take from parent1_only until exhausted, then take from intersection
            if i < len(parent1_only):
                baby[i] = parent1_only[i]
            else:
                baby[i] = intersection[intersection_count]
                intersection_count += 1
        else:
            # take from parent2_only until exhausted, then take from intersection
            if parent2_count < len(parent2_only):
                baby[i] = parent2_only[parent2_count]
                parent2_count += 1
            else:
                baby[i] = intersection[intersection_count]
                intersection_count += 1
    return np.sort(baby)


# needs additional parameter? for size of original input data
def mutate(parent, max):
    dimension_to_replace = np.random.choice(parent)
    index = list(parent).index(dimension_to_replace)
    rand_replacement = np.random.randint(max)  # (max+1) vs (max) depends on indexing
    while rand_replacement in parent:
        rand_replacement = np.random.randint(max)  # same as above comment
    parent[index] = rand_replacement
    return np.sort(parent)


# very basic - maybe should incorporate abs(sum(phi's)) as well
def EA_fitness(G_vector, error, age):
    age_cutoff = 5
    if age < age_cutoff:
        factor = 1
    else:
        factor = 1 / (age - (age_cutoff - 1))
    return np.linalg.norm(G_vector) * factor / (error + 1)


# Given phi loss function above
# Local Gradient
def EA_grad_phi(a, x, y):
    error = (f(a, x) - y)
    grad = 2 * error * grad_f(a, x)
    return [grad, error]

def EA_selection_singlea(L_k_list, G_list, age_list, error_list, n):
    fitnesses = dict()
    mapping = dict()
    count = 0
    for L_k in L_k_list:
        fitness = EA_fitness(G_list[count], age_list[count], error_list[count])
        fitnesses[count] = fitness  # [L_k,age_k,a_k] combination maps to a fitness
        mapping[count] = [list(np.sort(L_k)), age_list[count], G_list[count]]
        count += 1
    fitnesses_sorted = dict(sorted(fitnesses.items(), key=lambda item: item[1]))

    L_k_list_to_return = []
    age_list_to_return = []
    G_list_to_return = []
    count = 0
    for index in fitnesses_sorted.keys():
        count += 1
        if not mapping[index][0] in L_k_list_to_return:
            L_k_list_to_return.append(mapping[index][0])
            age_list_to_return.append(mapping[index][1] + 1)
            G_list_to_return.append(mapping[index][2])
        if len(L_k_list_to_return) > n - 1:
            break
    return [L_k_list_to_return, age_list_to_return, G_list_to_return]


def EA_breed_singlea(parent_vals, age_vals, L_max, split, num_children):
    parent_count = len(parent_vals)
    n = 0.1  # n is the number of parameters to mutate - 10% of points?
    children = []
    updated_population = parent_vals.copy()
    count = 0
    for i in range(parent_count):
        parent_vals[i] = list(parent_vals[i])
    for child in range(num_children):
        baby = parent_vals[0]  # not elegant - is needed to enter loop for first time
        while list(np.sort(baby)) in parent_vals:
            parent1_index = np.random.randint(parent_count)
            parent1 = parent_vals[parent1_index]
            decider = np.random.randn()
            # randomly choose between recombination+mutation or just mutation (weigh towards recombination+mutation)
            if decider > 0.2:
                # i.e. if recombination + mutation
                parent2 = parent_vals[np.random.randint(parent_count)]
                while list(parent1) == list(parent2):
                    # in case self-pollination
                    parent2 = parent_vals[np.random.randint(parent_count)]
                baby = recombine(parent1, parent2, split)
            else:
                # i.e. if only mutation
                baby = parent1
            baby = mutate(baby, L_max)
        count += 1
        updated_population.append(baby)
        age_vals.append(0)

    return [updated_population, age_vals]