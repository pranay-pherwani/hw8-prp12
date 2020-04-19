"""
hw8.py
Name(s):
NetId(s):
Date:
"""

import math
import random
import Organism as Org
import matplotlib.pyplot as plt

"""
crossover operation for genetic algorithm
"""
def crossover(parent1, parent2):
    # determine random index k
    k = random.randint(0,len(parent1)-1)
    # create child 1
    child1 = parent1[:k]+parent2[k:]
    # create child 2
    child2 = parent2[:k]+parent1[k:]
    return (child1, child2)

"""
mutation operation for genetic algorithm
"""
def mutation(genome, mutRate):
    # Flip the bit in the genome with chance mutRate
    for k in range(len(genome)):
        if random.random()<mutRate:
            genome[k]=1-genome[k]
    return genome

"""
selection operation for choosing a parent for mating from the population
"""
def selection(pop):
    # Pick a random between 0 and 1
    k = random.random()
    # Find the first org in pop to have accFit>k
    for org in pop:
        if org.accFit>k:
            return org
    # If none of them do, return the last org in pop
    return pop[-1]

"""
calcFit will calculate the fitness of an organism
"""
def calcFit(org, xVals, yVals):
    # Create a variable to store the running sum error.
    error = 0

    # Loop over each x value.
    for ind in range(len(xVals)):
        # Create a variable to store the running sum of the y value.
        y = 0
        
        # Compute the corresponding y value of the fit by looping
        # over the coefficients of the polynomial.
        for n in range(len(org.floats)):
            # Add the term c_n*x^n, where c_n are the coefficients.
            # Note: it is possible that squaring the number creates a value
            #       that is too large, i.e., an OverflowError. In this case,
            #       catch the error and treat the value as math.inf.
            try:
                y += org.floats[n] * (xVals[ind])**n
            except OverflowError:
                y += math.inf

        # Compute the squared error of the y values, and add to the running
        # sum of the error.
        # Note: it is possible that squaring the number creates a value
        #       that is too large, i.e., an OverflowError. In this case,
        #       catch the error and treat the value as math.inf.
        try:
            error += (y - yVals[ind])**2
        except OverflowError:
            error += math.inf

    # Now compute the sqrt(error), average it over the data points,
    # and return the reciprocal as the fitness.
    # Note that we have to check to make sure the fitness is not nan,
    # which means 'not a number'. If it is nan, then assign a fitness of 0.
    if error == 0:
        return math.inf
    else:
        fitness = len(xVals)/math.sqrt(error)
        if not math.isnan(fitness):
            return fitness
        else:
            return 0

"""
accPop will calculate the fitness and accFit of the population
"""
def accPop(pop, xVals, yVals):
    # Initialize total fitness
    totalFitness = 0
    # Calculate and set the fitness for each org in pop
    for org in pop:
        fit = calcFit(org,xVals,yVals)
        # Add the fitness to the total
        totalFitness+=fit
        org.fitness = fit
    # Sort the population in descending order of fitness
    pop.sort(reverse=True)
    # Calculate and set normalized fitness values for each org in pop
    for org in pop:
        org.normFit = org.fitness/totalFitness
    # Calculate accumulated fitness for each org in pop
    accum = 0
    for org in pop:
        accum+=org.normFit
        org.accFit=accum
    return pop

"""
initPop will initialize a population of a given size and number of coefficients
"""
def initPop(size, numCoeffs):
    # Get size-4 random organisms in a list.
    pop = [Org.Organism(numCoeffs) for x in range(size-3)]

    # Create the all 0s and all 1s organisms and append them to the pop.
    pop.append(Org.Organism(numCoeffs, [0]*(64*numCoeffs)))
    pop.append(Org.Organism(numCoeffs, [1]*(64*numCoeffs)))

    # Create an organism corresponding to having every coefficient as 1.
    bit1 = [0]*2 + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Create an organism corresponding to having every coefficient as -1.
    bit1 = [1,0] + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Return the population.
    return pop

"""
nextGeneration will create the next generation
"""
def nextGeneration(pop, numCoeffs, mutRate, eliteNum):
    return newPop

"""
GA will perform the genetic algorithm for k+1 generations (counting
the initial generation).

INPUTS
k:         the number of generations
size:      the size of the population
numCoeffs: the number of coefficients in our polynomials
mutRate:   the mutation rate
xVals:     the x values for the fitting
yVals:     the y values for the fitting
eliteNum:  the number of elite individuals to keep per generation
bestN:     the number of best individuals to track over time

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
fit:  the highest observed fitness value for each iteration
"""
def GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN):
    return (best,fit)

"""
runScenario will run a given scenario, plot the highest fitness value for each
generation, and return a list of the bestN number of top individuals observed.

INPUTS
scenario: a string to use for naming output files.
--- the remaining inputs are those for the call to GA ---

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
--- Plots are saved as: 'fit' + scenario + '.png' ---
"""
def runScenario(scenario, k, size, numCoeffs, mutRate, \
                xVals, yVals, eliteNum, bestN):

    # Perform the GA.
    (best,fit) = GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN)

    # Plot the fitness per generation.
    gens = range(k+1)
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(gens, fit)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.savefig('fitA.png', bbox_inches='tight')
    plt.close('all')

    # Return the best organisms.
    return best

"""
main function
"""
if __name__ == '__main__':

    # Flags to suppress any given scenario. Simply set to False and that
    # scenario will be skipped.
    scenA = True
    scenB = True
    scenC = True
    scenD = True
    
################################################################################
    ### Scenario A: Fitting to a constant function, y = 1. ###
################################################################################

    if scenA:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [1 for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'A'      # Set the scenario title.
        k = 100       # 100 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario B: Fitting to a constant function, y = 5. ###
################################################################################
    
    if scenB:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [5 for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'B'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario C: Fitting to a quadratic function, y = x^2 - 1. ###
################################################################################
    
    if scenC:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = x^2 - 1 corresponding to the x values.
        yVals = [x**2-1 for x in xVals]

        # Set the other parameters for the GA.
        sc = 'C'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()

################################################################################
    ### Scenario D: Fitting to a quadratic function, y = cos(x). ###
################################################################################
    
    if scenD:
        # Create the x values ranging from -5 to 5 with a step of 0.1.
        xVals = [0.1*n-5 for n in range(101)]

        # Create the y values for y = cos(x) corresponding to the x values.
        yVals = [math.cos(x) for x in xVals]

        # Set the other parameters for the GA.
        sc = 'D'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 5 # Quartic polynomial with 4 zeros!
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()