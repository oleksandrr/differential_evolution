#!/usr/bin/python

from __future__ import division

__all__ = ['mutate', 'crossover', 'select', 'minimize']

import numpy; from numpy import (array, random)

def mutate(original_population, original_population_costs, F, method="rand/1"):
    # In some cases random_unique_indices can become a bottleneck for the whole algorithm,
    # so further attention may be warranted. Circumstantially, it is found that the
    # uniqueness constraint on the parents is non-critical, so in principle one might
    # wish to choose the parents without restrictions for better mutation performance.
    def random_unique_indices(source_length, count, inclusions=[], exclusions=[]):
        indices = list(inclusions)
        while len(indices) < count:
            next = random.randint(source_length)
            while next in indices or next in exclusions:
                next = random.randint(source_length)
            indices.append(next)
        return indices
    
    original_population = array(original_population,copy=False)
    original_population_costs = array(original_population_costs,copy=False)
    
    population_size = len(original_population)
    differential_evolution_indices = lambda count: array(
     [ random_unique_indices(population_size, count, exclusions=[index])
       for index in xrange(population_size) ]
    )
    
    if method == "MDE5":
        # Based on Thangaraj et al., Appl. Math. Comp. 216 (2), 532 (2010).
        best_individual = original_population[numpy.argmin(original_population_costs)]
        parents = original_population[differential_evolution_indices(2)]
        L = random.laplace(F, F/4, (population_size,1))
        return parents[:,0] + L*(best_individual - parents[:,1])
    elif method == "current-to-best/1":
        # Based on the current-to-p-best/1 method of JADE as presented by Jinqiao Zhang and
        # A. C. Sanderson in IEEE Trans. Evol. Comput. 13 (5), 945 (2009). Here we pick from
        # the first decile, but this proportion is arbitrary and may be varied.
        p = min(1, int(round(0.1*population_size)))
        p_best_individuals = original_population[numpy.argsort(original_population_costs)[:p]]
        random_p_best = p_best_individuals[random.randint(p, size=population_size)]
        parents = original_population[differential_evolution_indices(2)]
        return original_population + F*(
         random_p_best - original_population + parents[:,0] - parents[:,1]
        )
    elif method == "rand/2/dir":
        # This method behaves similarly to the reflection operator of the Nelder-Mead algorithm.
        # It is not particularly effective for low-dimensional problems, but performs well on
        # high-dimensional non-separable functions.
        # TODO: A more intelligent hybridization of DE and N-M is probably desirable.
        indices = differential_evolution_indices(5)
        sorted_indices = indices[ numpy.arange(population_size)[:,None],
                                  numpy.argsort(original_population_costs[indices]) ]
        parents = original_population[sorted_indices]
        return parents[:,0] + F*(parents[:,1] + parents[:,2] - parents[:,3] - parents[:,4])
    elif method == "best/1/dither":
        # The classical best/1 mutator strongly favours local search; here we add dither to
        # help avoid misconvergence by improving the diversity of the mutants.
        best_individual = original_population[numpy.argmin(original_population_costs)]
        parents = original_population[differential_evolution_indices(2)]
        N = random.normal(F, F/2, (population_size,1))
        return best_individual + N*(parents[:,0] - parents[:,1])
    elif method == "best/2":
        best_individual = original_population[numpy.argmin(original_population_costs)]
        parents = original_population[differential_evolution_indices(4)]
        return best_individual + F*(parents[:,0] + parents[:,1] - parents[:,2] - parents[:,3])
    else: # invalid or no method specified; use default method == "rand/1"
        parents = original_population[differential_evolution_indices(3)]
        return parents[:,0] + F*(parents[:,1] - parents[:,2])


# Here we can safely rely on mutant_population being an array, because it is produced in this
# form by mutate(). However, the first-generation original_population might be passed as a list,
# so we must wrap it in array() as a precaution. original_population and mutant_population will
# however have the same size and shape regardless of this potential type difference.
#
def crossover(original_population, mutant_population, C, method="binomial"):
    
    original_population = array(original_population,copy=False)
    
    if method == "arithmetic":
        # TODO: This method is questionably effective. A different (rotationally
        #       invariant) one should be implemented instead if possible.
        return original_population + C*(mutant_population - original_population)
    else: # invalid or no method specified; use default method == "binomial"
        return numpy.where(
         random.random_sample(mutant_population.shape) < C,
         mutant_population,
         original_population
        )


# The size of the population need not be conserved from one generation to the next, so we may
# produce a final population by selecting n individuals with any n such that the uniqueness constraint
# on the parents is satisfiable (thus avoiding an infinite loop), i.e.:
#  n > 2 (MDE5, current-to-best/1, best/1/dither)
#  n > 3 (rand/1).
#  n > 4 (best/2)
#  n > 5 (rand/2/dir)
#
def select(objective_function,
           original_population, original_population_costs,
           trial_population, method="Storn-Price", stochastic=False,
           fitness_monitor=None, generation_monitor=None):
    
    original_population = array(original_population,copy=False)
    original_population_costs = array(original_population_costs,copy=False)
    
    # Here the objective function is evaluated for each trial vector. For brevity's sake, we will call
    # the objective function value associated with each vector its "cost" (which is to be minimized).
    #
    trial_population_costs = array(
     map(objective_function, trial_population)
    )
    
    if method == "elitist":
         all_costs = numpy.concatenate((original_population_costs, trial_population_costs))
         indices = numpy.argsort(all_costs)[:len(all_costs)//2]
         
         new_population_costs = all_costs[indices]
         new_population = numpy.concatenate((original_population, trial_population))[indices]
    else: # invalid or no method specified; use default method == "Storn-Price"
         mask = original_population_costs < trial_population_costs
         
         new_population_costs = numpy.where(mask, original_population_costs, trial_population_costs)
         new_population = numpy.where(mask.reshape((-1,1)), original_population, trial_population)
    
    # For stochastic objective functions it is unhelpful to have the function values recorded for each
    # parameter vector once and for all, because vectors with small objective function values obtained
    # by chance tend to proliferate and disrupt the optimization process. Here we (re-)evaluate the
    # objective function if necessary in order to avoid this. Note that this option should only be
    # used for genuinely stochastic objective functions as it doubles the number of function
    # evaluations required at each generation.
    #
    if stochastic:
        new_population_costs = array(
         map(objective_function, new_population)
        )
    
    # generation_monitor() is a function that can receive a tuple of the form (array of objective
    # function values, array of parameter vectors) at each iteration. This allows results to be
    # displayed or written out to disk as the optimization proceeds, if required.
    #
    if generation_monitor is not None:
        generation_monitor((new_population_costs, new_population))
    
    # fitness_monitor() is a function that can receive an array of objective function values at each
    # iteration. This is intended to allow the algorithm to determine its rate of improvement over
    # successive generations and so to stop when the optimization has converged or if the population
    # has stagnated.
    #
    if fitness_monitor is not None:
        fitness_monitor(new_population_costs)
    
    return (new_population_costs, new_population)


def minimize(objective_function, initial_population, F, C,
             convergence_tolerance, convergence_history_length, max_unconverged_iterations,
             mutation_method="rand/1", crossover_method="binomial",
             selection_method="Storn-Price", stochastic=False,
             output_function=None):
    
    if not (0.0 < F and F <= 2.0):
        raise ValueError(
         "Inappropriate value of differential mixing strength F. Suitable values are 0.0 < F <= 2.0."
        )
    
    if not (0.0 < C and C <= 1.0):
        raise ValueError(
         "Invalid value of crossover probability C. Allowable values are 0.0 < C <= 1.0."
        )
    
    if convergence_history_length < 2:
        raise ValueError(
         "Insufficient convergence history length. At least 2 generations are needed to assess convergence."
        )
    
    if mutation_method not in (
     # Classical methods
       "rand/1", "best/2",
     # Modified classical methods
       "best/1/dither", "rand/2/dir",
     # Non-classical methods
       "current-to-best/1", "MDE5" 
    ):
        raise ValueError(
         "Unknown mutation method. Options are " +
         "\"rand/1\" (default), \"best/2\", " +
         "\"best/1/dither\", \"rand/2/dir\", " +
         "\"current-to-best/1\", or \"MDE5\"."
        )
    
    if crossover_method not in ("binomial", "arithmetic"):
        raise ValueError(
         "Unknown crossover method. Options are \"binomial\" (default) or \"arithmetic\"."
        )
        
    if selection_method not in ("Storn-Price", "elitist"):
        raise ValueError(
         "Unknown selection method. Options are \"Storn-Price\" (default) or \"elitist\"."
        )
    
    # convergence_monitor() is a coroutine that maintains an internal history of objective function values
    # over a specified number of generations. It functions as an iterator that terminates when the cumulative
    # improvement over this number of generations falls below a given tolerance or when a specified maximum
    # number of iterations is exceeded without achieving convergence. The minimum number of iterations
    # performed is equal to the lesser of convergence_history_length or max_unconverged_iterations.
    #
    def convergence_monitor(convergence_tolerance, convergence_history_length, max_unconverged_iterations):
        history = numpy.empty(convergence_history_length,dtype=numpy.float64); history[:] = numpy.nan
        mean_abs_difference = numpy.float64(numpy.nan)
        
        epsilon = numpy.finfo(numpy.float64).eps
        
        iteration_count = 1
        while iteration_count <= max_unconverged_iterations:
            new_fitness = yield iteration_count
            if new_fitness is not None:
                 history[0] = new_fitness; history = numpy.roll(history, -1)
                 mean_abs_difference = numpy.mean(
                  numpy.abs(numpy.diff(history))
                 )
            else:
                iteration_count += 1
                if mean_abs_difference < convergence_tolerance + epsilon * mean_abs_difference:
                    return
    
    # The convergence monitor must be instantiated and initialised before use. Cost values are then recorded 
    # for each generation by setting the fitness_monitor parameter of select() to a function that accepts a
    # list of objective function values for a generation, reduces the list to a single number, and calls the
    # send() method with this number as a parameter: here this is done by update_cm().
    #
    cm = convergence_monitor(convergence_tolerance, convergence_history_length, max_unconverged_iterations)
    cm.send(None)
    
    update_cm = lambda costs: cm.send(numpy.mean(costs))
    
    population = array(initial_population,copy=False)
    costs = map(objective_function, population)
    update_cm(costs)
    if output_function is not None:
        output_function((costs, population))
    
    # Since the initial population was taken to constitute the first generation, this loop will actually
    # perform at most (max_unconverged_iterations - 1) further iterations if convergence is not achieved.
    #
    for iterations in cm:
        (costs, population) = select(
          objective_function,
          population, costs,
          crossover(
           population,
           mutate(population, costs, F, method=mutation_method),
           C, method=crossover_method
          ), method=selection_method, stochastic=stochastic,
          fitness_monitor=update_cm, generation_monitor=output_function
         )
    
    return (iterations, (costs, population))


