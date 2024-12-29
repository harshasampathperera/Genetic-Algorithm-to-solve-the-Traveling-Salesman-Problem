# Genetic-Algorithm-to-solve-the-Traveling-Salesman-Problem

import numpy as np
import random
import string


# Function to generate random chromosome with random keys
def generate_chromosome(num_cities):
    return np.random.rand(num_cities)


# Fitness function: Calculate total distance of the route
def fitness(chromosome, dist_matrix):
    # Convert the chromosome (random keys) into a sorted order of cities
    order = np.argsort(chromosome)
    total_distance = 0
    for i in range(len(order) - 1):
        total_distance += dist_matrix[order[i], order[i + 1]]
    # Add distance from last city back to the first city
    total_distance += dist_matrix[order[-1], order[0]]
    return total_distance


# Selection: Tournament selection
def tournament_selection(population, fitness_values, tournament_size=3):
    selected = np.random.choice(range(len(population)), tournament_size, replace=False)
    best_index = selected[np.argmin([fitness_values[i] for i in selected])]
    return population[best_index]


# Crossover using random key method with probability
def crossover(parent1, parent2, p):
    child = np.zeros(len(parent1))
    for i in range(len(parent1)):
        if random.random() < p:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child


# Mutation: swap two random cities
def mutate(chromosome):
    idx1, idx2 = np.random.choice(len(chromosome), 2, replace=False)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]


# Genetic Algorithm
def genetic_algorithm(
    num_cities,
    dist_matrix,
    num_generations,
    population_size,
    mutation_rate,
    crossover_prob,
):
    # Initialize population
    population = [generate_chromosome(num_cities) for _ in range(population_size)]

    for generation in range(num_generations):
        # Calculate fitness for each chromosome
        fitness_values = [fitness(chromosome, dist_matrix) for chromosome in population]

        # Create a new population
        new_population = []
        for _ in range(population_size // 2):  # Two children per iteration
            # Selection
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            # Crossover
            child1 = crossover(parent1, parent2, crossover_prob)
            child2 = crossover(parent2, parent1, crossover_prob)

            # Mutation
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    # Find the best solution in the final population
    fitness_values = [fitness(chromosome, dist_matrix) for chromosome in population]
    best_chromosome = population[np.argmin(fitness_values)]
    best_route = np.argsort(best_chromosome)

    return best_route, fitness_values[np.argmin(fitness_values)]


# Input the distance matrix from the user
def input_distance_matrix(num_cities):
    print("\nEnter the distance between each pair of cities:")
    dist_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = float(
                input(f"Distance between city {chr(65 + i)} and {chr(65 + j)}: ")
            )
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance  # Since the matrix is symmetric

    return dist_matrix


# Example usage
if __name__ == "__main__":
    # Input Parameters
    num_cities = int(input("Enter the number of cities: "))  # Number of cities
    num_generations = int(
        input("Enter the number of generations (iterations): ")
    )  # Number of iterations
    population_size = int(
        input("Enter the population size: ")
    )  # Size of the population
    mutation_rate = float(
        input("Enter the mutation rate (e.g., 0.1): ")
    )  # Probability of mutation
    crossover_prob = float(
        input("Enter the crossover probability p (e.g., 0.5): ")
    )  # Probability to inherit gene from parent 1

    # City labels: A, B, C, ...
    city_labels = [chr(65 + i) for i in range(num_cities)]

    # Input distance matrix from the user
    dist_matrix = input_distance_matrix(num_cities)

    # Print the distance matrix
    print("\nDistance Matrix:")
    print(dist_matrix)

    # Run the Genetic Algorithm
    best_route, shortest_distance = genetic_algorithm(
        num_cities,
        dist_matrix,
        num_generations,
        population_size,
        mutation_rate,
        crossover_prob,
    )

    # Print results
    best_route_labels = [city_labels[i] for i in best_route]
    print(f"\nBest route: {' -> '.join(best_route_labels)}")
    print(f"Shortest distance: {shortest_distance}")
