import random
import numpy as np

# Define the items
items = [
    {"name": "Car Key", "value": 300000, "volume": 0.05, "quantity": 1},
    {"name": "Laptop", "value": 15000, "volume": 1.75, "quantity": 2},
    {"name": "Watch", "value": 600, "volume": 0.08, "quantity": 4},
    {"name": "Jewelry", "value": 600, "volume": 0.015, "quantity": 12},
    {"name": "Cellphone", "value": 9000, "volume": 0.2, "quantity": 20},
    {"name": "Tablet", "value": 15000, "volume": 0.6, "quantity": 10}
]

# Constants
BAG_CAPACITY = 10  # 10L
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.05
GENERATIONS = 10000
TOURNAMENT_SIZE = 3

def create_individual():
    return np.random.randint(2, size=sum(item['quantity'] for item in items))

def combine(parent_a, parent_b, c_rate):
    if random.random() <= c_rate:
        c_point = np.random.randint(1, len(parent_a))
        offspring_a = np.concatenate((parent_a[:c_point], parent_b[c_point:]))
        offspring_b = np.concatenate((parent_b[:c_point], parent_a[c_point:]))
    else:
        offspring_a, offspring_b = np.copy(parent_a), np.copy(parent_b)
    return offspring_a, offspring_b

def mutate(individual, m_rate):
    for i in range(len(individual)):
        if random.random() <= m_rate:
            individual[i] = 1 - individual[i]
    return individual

def evaluate(individual):
    total_value = 0
    total_volume = 0
    idx = 0
    for item in items:
        for _ in range(item['quantity']):
            if individual[idx] == 1:
                total_value += item['value']
                total_volume += item['volume']
            idx += 1
    
    if total_volume > BAG_CAPACITY:
        return 0  # Invalid solution
    return total_value

def select(population, evaluation, tournament_size):
    tournament = random.sample(range(len(population)), tournament_size)
    return population[max(tournament, key=lambda i: evaluation[i])]

def genetic_algorithm(population_size, c_rate, m_rate, generations):
    population = [create_individual() for _ in range(population_size)]
    best_individual = None
    best_evaluation = float('-inf')
    past_evaluation = 0
    count = 0
    n_generations = 0  
    for gen in range(generations):
        evaluation = [evaluate(ind) for ind in population]

        if abs(max(evaluation) - past_evaluation)/max(evaluation) < 0.005:
            count += 1
        else:
            count = 0
        
        if count == 10:
            n_generations = gen
            break

        past_evaluation = max(evaluation)


        if max(evaluation) > best_evaluation:
            best_index = evaluation.index(max(evaluation))
            best_individual = population[best_index]
            best_evaluation = evaluation[best_index]

        

        new_population = []
        for _ in range(population_size // 2):
            parent_a = select(population, evaluation, TOURNAMENT_SIZE)
            parent_b = select(population, evaluation, TOURNAMENT_SIZE)
            offspring_a, offspring_b = combine(parent_a, parent_b, c_rate)
            new_population.extend([mutate(offspring_a, m_rate), mutate(offspring_b, m_rate)])
        
        population = new_population

    return best_individual, best_evaluation, n_generations

def decode_solution(solution):
    selected_items = []
    idx = 0
    for item in items:
        for _ in range(item['quantity']):
            if solution[idx] == 1:
                selected_items.append(item['name'])
            idx += 1
    return selected_items

# Run the genetic algorithm
best_solution, best_value, n_generations = genetic_algorithm(POPULATION_SIZE, CROSSOVER_RATE, MUTATION_RATE, GENERATIONS)

print(f"Best value: ${best_value}")
print(f"Number of generations: {n_generations}")
print("Selected items:", decode_solution(best_solution))
print("Total volume:", sum(item['volume'] for item in items for _ in range(item['quantity']) if best_solution[sum(it['quantity'] for it in items[:items.index(item)]) + _] == 1))