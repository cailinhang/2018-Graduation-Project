"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat2.reporting import ReporterSet
from neat2.math_util import mean
from neat2.six_util import iteritems, itervalues
from neat2.checkpoint import Checkpointer
from neat2.parallel import ParallelEvaluator

class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        
        self.checkpointer = Checkpointer(generation_interval=3,
                                         time_interval_seconds=None,
                                         filename_prefix='./checkpoints/parallel-group-4000-train-neat-checkpoint-')
        self.reporters = ReporterSet()
        self.config = config
        self.has_eval = {}
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        
        if initial_state is None:
            self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state
            genomes = list(iteritems(self.population))            
            start_idx=0
            for genome_id, genome in genomes:
                self.has_eval[genome_id] = 1
                if genome_id > start_idx:
                    start_idx = genome_id                     
            self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation, start_idx + 1)

        self.best_genome = []

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        parallel = ParallelEvaluator(num_workers=5, eval_function=fitness_function)
    
        if self.generation > 0: # start from saved_state 
            
            self.reporters.start_generation(self.generation)
            
            # Gather and report statistics.
            best = None
            
            for g in itervalues(self.population):        
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            self.best_genome.append((best, best.fitness))
            
            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
                
            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)
                               
            self.generation += 1    

        k = 0
        while n is None or k < n:
            k += 1
            self.checkpointer.start_generation(self.generation) 
            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            #fitness_function(list(iteritems(self.population)), self.config)
            
            genomes_l = []
            
            for genome_id, genome in list(iteritems(self.population)):
                
                if self.has_eval.__contains__(genome_id):
                    continue
                else:
                    self.has_eval[genome_id] = 1
                    genomes_l.append((genome_id, genome))
                    
            #genomes = parallel.evaluate(list(iteritems(self.population)), self.config)
            genomes = parallel.evaluate(genomes_l, self.config)
            #print('---')
            for genome_id, genome in genomes:
                #print(genome_id, genome.fitness)
                self.population[genome_id] = genome

            self.species.speciate(self.config, self.population, self.generation)                
            
                            
            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            #if self.best_genome is None or best.fitness > self.best_genome.fitness:
                #self.best_genome = best
            self.best_genome.append((best, best.fitness))

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    #print(fv.item(),' >= ',  self.config.fitness_threshold)
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            if self.generation - self.checkpointer.last_generation_checkpoint >= \
                self.checkpointer.generation_interval:
                print('save checkpoint')
                self.checkpointer.end_generation(self.config, self.population, self.species)
                            
                

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
