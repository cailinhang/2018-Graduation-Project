"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        #print(genomes)
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (ignored_genome_id, genome, config)))
        #print('ok')            
        # assign the fitness back to each genome
        j = 0
        genomes_l = []
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            j +=1
            #genome.fitness = job.get(timeout=self.timeout)
            genome = job.get(timeout=self.timeout)            
            genomes_l.append((ignored_genome_id, genome))
            print('{0}: {1:3.3f} {2}'.format(j, genome.fitness, ignored_genome_id))
        #print(genomes)            
        #print('*****')
        #print(genomes_l)
#        for genome_id, genome in genomes:            
#            print(genome_id, genome.fitness)
#        for genome_id, genome in genomes_l:            
#            print(genome_id, genome.fitness)            
        return genomes_l            