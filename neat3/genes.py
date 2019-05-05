"""Handles node and connection genes."""
import warnings
from random import random
from neat3.attributes import FloatAttribute, BoolAttribute, StringAttribute, ListAttribute, ListAttribute1

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        assert isinstance(self.key,type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key,other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__,cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        if isinstance(self, DefaultNodeGene):
             # 这里对 new_gene.layer赋值
            new_gene = self.__class__(self.key, self.layer)
        else:            
            new_gene = self.__class__(self.key)         
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        # Add    
        if isinstance(self, DefaultNodeGene):  
            
            new_gene.in_nodes = self.in_nodes.copy()
#            assert len(new_gene.kernal) == len(self.kernal)
            
            new_gene.kernal = self.kernal.copy() # TODO: add
            new_gene.kernal1 = self.kernal1.copy() # TODO: add
#            if len(new_gene.in_nodes) != len(self.in_nodes):   
#                print(new_gene.in_nodes,' ', self.in_nodes)
            assert len(new_gene.in_nodes) == len(self.in_nodes)                
        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        if isinstance(self, DefaultNodeGene):
            assert self.layer == gene2.layer
            new_gene = self.__class__(self.key, self.layer)
            new_gene.kernal = self.kernal.copy() #
            new_gene.kernal1 = self.kernal1.copy() #
            new_gene.in_nodes = self.in_nodes.copy() # 
            #print('in_nodes' , self.in_nodes)
        else:
            new_gene = self.__class__(self.key)

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.        
        for a in self._gene_attributes:
            if random() < 0.5:
                if (a.name == 'kernal') or (a.name == 'kernal1'):
                    pass # TODO: 修改#                   
                else:
                    setattr(new_gene, a.name, getattr(self, a.name))
            else:
                if (a.name == 'kernal') or (a.name == 'kernal1'):
                    pass # TODO: 修改                   
                else:
                    setattr(new_gene, a.name, getattr(gene2, a.name))
                    
            # Newly Modified, originally, random()< 0.2                    
            if (random() < 0.2):
                if ((a.name == 'weight') or (a.name == 'bias')):
                    lamda = random()
                    tmpa = getattr(self, a.name)
                    tmpb = getattr(gene2, a.name)
                    # Newly Modified
                    if lamda < 0.05:
                        tmp = (1 +lamda) * tmpa
                    elif lamda < 0.1:
                        tmp = (0.75 + lamda) * tmpb                        
                    else:                        
                        tmp = tmpa * lamda + tmpb * (1 - lamda)
                    setattr(new_gene, a.name, tmp)
                #TODO: 5 should be a config parameter
                elif (a.name == 'kernal') and self.layer % 2 ==0:
#                    pass
                     for i in range(len(self.in_nodes)):
                         #print(i, ' ', self.in_nodes[i])
                         
                         for j in range(len(gene2.in_nodes)):
                             
                             if gene2.in_nodes[j] == self.in_nodes[i]:
                                 #print(j, ' ', self.in_nodes[i])
                                 for k in range(len(self.kernal[0])):
                                     
                                     lamda = random()
                                     # Amplify  
                                     #TODO: Note that the boundary of kernal
                                     # is not limited
                                     if lamda < 0.05:
                                         new_gene.kernal[i][k] = \
                                         (1 + lamda)* self.kernal[i][k]
                                     elif lamda < 0.1:
                                         new_gene.kernal[i][k] = \
                                         (0.75 + lamda)* gene2.kernal[j][k]
                                         
#                                     if lamda < 0.5:
#                                         lamda = 1 - lamda
                                     else:
                                         #print(i, ' ',len(self.kernal), ' k = ', k)                                                                                    
                                         new_gene.kernal[i][k] = \
                                             lamda * self.kernal[i][k] + \
                                             (1 - lamda) * gene2.kernal[j][k]
                    # TODO: 修改                    
#                    for in_channel in range(len(new_gene.kernal)):
#                        # Now, len(kernel[0]) is kernel_size= 9
#                        for i in range(len(new_gene.kernal[0])): 
#                            lamda = random()
#                            tmpa = self.kernal[in_channel][i]
#                            tmpb = gene2.kernal[in_channel][i]
#                            tmp = tmpa * lamda + tmpb * (1 - lamda)
#                            new_gene.kernal[in_channel][i] = tmp
                elif (a.name == 'kernal1') and self.layer % 2 ==1:
#                    pass
                     #print('self.layer ', self.layer, ' self.in_nodes ' , self.in_nodes)
                     #print(' gene2.in_nodes ', gene2.in_nodes)
                     for i in range(len(self.in_nodes)):
                         #print(i, ' ', self.in_nodes[i])
                         
                         for j in range(len(gene2.in_nodes)):
                             
                             if gene2.in_nodes[j] == self.in_nodes[i]:
                                 #print(j, ' ', self.in_nodes[i])
                                 #print(len(self.kernal1[0]))
                                 for k in range(len(self.kernal1[0])):
                                     #print('k', k)
                                     lamda = random()
                                     # Amplify  
                                     #TODO: Note that the boundary of kernal
                                     # is not limited
                                     if lamda < 0.05:
                                         new_gene.kernal1[i][k] = \
                                         (1 + lamda)* self.kernal1[i][k]
                                     elif lamda < 0.1:
                                         new_gene.kernal1[i][k] = \
                                         (0.75 + lamda)* gene2.kernal1[j][k]
                                         
#                                     if lamda < 0.5:
#                                         lamda = 1 - lamda
                                     else:
                                         #print('i ' , i, 'new_gene ',len(new_gene.kernal1), ' k = ', k)                                                                                    
                                         #if len(self.kernal1) == 1:
                                             #print(self.kernal1)
                                         new_gene.kernal1[i][k] = \
                                             lamda * self.kernal1[i][k] + \
                                             (1 - lamda) * gene2.kernal1[j][k]


        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum') ,
                        ListAttribute('kernal') ,
                        ListAttribute1('kernal1')
                        ]

    def __init__(self, key, layer ):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)        
        
        self.in_nodes = [] # 卷积层结点的 输入channels
        # Added by Andrew
        self.layer = layer

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient

