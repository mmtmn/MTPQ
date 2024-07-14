import threading
import queue
import random
import time

class Task:
    """A simple Task class with a name and a priority. The process method simulates task processing."""
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

    def process(self):
        # Simulate task processing time
        time.sleep(random.uniform(0.1, 0.5))
        print(f"Processed {self.name} with priority {self.priority}")

    def __lt__(self, other):
        """Define a less-than method to compare tasks based on priority."""
        return self.priority < other.priority


def add_tasks_to_queue(q, num_tasks=20):
    """Populate the priority queue with a specified number of tasks with random priorities."""
    for i in range(num_tasks):
        task = Task(f"Task-{i}", random.randint(1, 10))
        q.put((task.priority, task))  # The queue uses the first tuple element as the priority

def process_tasks_from_queue(q):
    """Worker function for threads to process tasks from the queue."""
    while not q.empty():
        try:
            _, task = q.get(timeout=1)
            task.process()
            q.task_done()
        except queue.Empty:
            break

def evaluate_strategy(num_workers):
    """Evaluate a strategy by timing how long it takes to process all tasks using a given number of threads."""
    pq = queue.PriorityQueue()
    add_tasks_to_queue(pq)

    start_time = time.time()
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(target=process_tasks_from_queue, args=(pq,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    total_time = time.time() - start_time
    return total_time

def genetic_algorithm(population, generations=10):
    """Run the genetic algorithm to evolve the number of worker threads."""
    for generation in range(generations):
        # Evaluate all individuals in the population
        fitness_scores = [(evaluate_strategy(individual), individual) for individual in population]
        # Sort by fitness (lower time is better)
        fitness_scores.sort()

        # Select the best half of the population or at least 2 individuals
        population = [individual for _, individual in fitness_scores[:max(len(population)//2, 2)]]

        # Ensure there's enough population for crossover
        if len(population) < 2:
            population += [random.randint(1, 10) for _ in range(2 - len(population))]

        # Crossover and mutation: randomly pair and average, then mutate by adding/subtracting threads
        next_gen = []
        while len(next_gen) < len(fitness_scores):
            if len(population) > 1:
                parent1, parent2 = random.sample(population, 2)
                child = (parent1 + parent2) // 2
                # Mutation chance
                if random.random() < 0.1:
                    child += random.choice([-1, 1])
                    child = max(1, child)  # Ensure at least one thread
                next_gen.append(child)
            else:
                # If not enough parents, repeat the existing or mutate them
                child = population[0] + random.choice([-1, 1])
                child = max(1, child)  # Ensure at least one thread
                next_gen.append(child)

        population = next_gen
        print(f"Generation {generation + 1}: Best time {fitness_scores[0][0]} with {fitness_scores[0][1]} threads")
    return population


# Initial population and run the genetic algorithm
initial_population = [random.randint(1, 10) for _ in range(10)]
best_strategies = genetic_algorithm(initial_population)
print("Best strategies:", best_strategies)
