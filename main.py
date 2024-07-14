import threading
import queue
import random
import time

class Task:
    """A task class with name, priority, and complexity attributes. Complexity affects processing time."""
    def __init__(self, name, priority, complexity):
        self.name = name
        self.priority = priority
        self.complexity = complexity

    def process(self):
        # Processing time is influenced by complexity
        time.sleep(self.complexity * random.uniform(0.1, 0.5))
        print(f"Processed {self.name} with priority {self.priority} and complexity {self.complexity}")

    def __lt__(self, other):
        """Define a less-than method to compare tasks based on priority."""
        return self.priority < other.priority

memory = {}

def add_tasks_to_queue(q, context):
    """Populate the priority queue with tasks based on the provided context (e.g., 'high_stress')."""
    if context == 'high_stress':
        num_tasks = 30
        complexities = [random.uniform(1, 3) for _ in range(num_tasks)]
    else:
        num_tasks = 15
        complexities = [random.uniform(0.5, 1.5) for _ in range(num_tasks)]

    for i in range(num_tasks):
        priority = random.randint(1, 10)
        complexity = complexities[i]
        task = Task(f"Task-{i}", priority, complexity)
        q.put((task.priority, task))

def process_tasks_from_queue(q):
    """Process tasks from the queue, adjusting for memory of past tasks."""
    while not q.empty():
        try:
            _, task = q.get(timeout=1)
            if task.name in memory:
                task.complexity *= 0.9  # Reduce complexity if task is remembered
            task.process()
            memory[task.name] = 'processed'  # Store task in memory
            q.task_done()
        except queue.Empty:
            break

def evaluate_strategy(num_workers, context):
    """Evaluate a strategy based on the number of workers and context."""
    pq = queue.PriorityQueue()
    add_tasks_to_queue(pq, context)

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

def genetic_algorithm(population, context, generations=10):
    """Genetic algorithm to evolve the number of worker threads."""
    for generation in range(generations):
        fitness_scores = [(evaluate_strategy(individual, context), individual) for individual in population]
        fitness_scores.sort()

        population = [individual for _, individual in fitness_scores[:max(len(population)//2, 2)]]

        if len(population) < 2:
            population += [random.randint(1, 10) for _ in range(2 - len(population))]

        next_gen = []
        while len(next_gen) < len(fitness_scores):
            if len(population) > 1:
                parent1, parent2 = random.sample(population, 2)
                child = (parent1 + parent2) // 2
                if random.random() < 0.1:
                    child += random.choice([-1, 1])
                    child = max(1, child)
                next_gen.append(child)
            else:
                child = population[0] + random.choice([-1, 1])
                child = max(1, child)
                next_gen.append(child)

        population = next_gen
        print(f"Generation {generation + 1}: Best time {fitness_scores[0][0]} with {fitness_scores[0][1]} threads")
    return population

# Initialize and run the genetic algorithm with a given context
initial_population = [random.randint(1, 10) for _ in range(10)]
context = 'high_stress'  # Can be changed to 'normal' or other contexts
best_strategies = genetic_algorithm(initial_population, context)
print("Best strategies:", best_strategies)
