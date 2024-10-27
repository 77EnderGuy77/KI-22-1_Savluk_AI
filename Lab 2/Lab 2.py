import numpy as np
import sys
import time

def eval_func(chromosome):
    x, y, z = chromosome
    value = 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)
    return value,

def initialize_population(pop_size, chromosome_length, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, chromosome_length))

def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        competitors_indices = np.random.choice(population.shape[0], 2, replace=False)
        competitors = population[competitors_indices]

        parent = competitors[np.argmax([fitness_scores[i] for i in competitors_indices])]
        parents.append(parent)
    return np.array(parents)

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size[0]):
        parent1_idx = np.random.randint(0, parents.shape[0])
        parent2_idx = np.random.randint(0, parents.shape[0])
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        crossover_point = np.random.randint(1, parents.shape[1] - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
    return np.array(offspring)

def mutate(offspring, mutation_rate, bounds):
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            mutation_value = np.random.uniform(bounds[0], bounds[1])
            mutation_index = np.random.randint(0, offspring.shape[1])
            offspring[i][mutation_index] = mutation_value
    return offspring

def display_loading_animation(iteration, total_iterations):
    symbols = ['⠋', '⠙', '⠸', '⠼', '⠴', '⠶', '⠦', '⠧', '⠇', '⠏']
    symbol = symbols[iteration % len(symbols)]
    sys.stdout.write(f'\rProcessing generation {iteration + 1}/{total_iterations} {symbol}')
    sys.stdout.flush()

def genetic_algorithm(pop_size, num_generations, chromosome_length, bounds, num_parents, mutation_rate):
    population = initialize_population(pop_size, chromosome_length, bounds)
    
    for generation in range(num_generations):
        fitness_scores = np.array([eval_func(chromosome)[0] for chromosome in population])
        
        parents = select_parents(population, fitness_scores, num_parents)
        
        offspring_size = (pop_size - parents.shape[0], chromosome_length)
        offspring = crossover(parents, offspring_size)
        
        offspring = mutate(offspring, mutation_rate, bounds)
        
        population = np.vstack((parents, offspring))
        
        display_loading_animation(generation, num_generations)

    fitness_scores = np.array([eval_func(chromosome)[0] for chromosome in population])
    best_chromosome = population[np.argmax(fitness_scores)]
    best_score = np.max(fitness_scores)
    
    return best_chromosome, best_score

# Настройки параметров
pop_size = 1000          # Розмір популяції
num_generations = 1000   # Кількість поколінь
chromosome_length = 3    # Довжина хромосоми
bounds = (-10, 10)       # Границы для x, y, z
num_parents = 50         # Кількість батьків
mutation_rate = 0.1      # Ймовірність мутації

# Запуск генетичного алгоритма з виміром часу
start_time = time.time()  # Починаємо відлік часу

best_chromosome, best_score = genetic_algorithm(pop_size, num_generations, chromosome_length, bounds, num_parents, mutation_rate)

end_time = time.time()  # Закінчуємо відлік часу
total_time = end_time - start_time  # Обчислюємо час виконання

print(f"\nЛучшая хромосома: {best_chromosome}")
print(f"Лучший результат функции: {best_score}")
print(f"\nЧас виконання: {total_time:.2f} секунд")