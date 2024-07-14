import pygame
import sys
import threading
from main import add_tasks_to_queue, process_tasks_from_queue, Task, queue

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Task Processing Simulation")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Setup task queue and threading
task_queue = queue.PriorityQueue()
thread = threading.Thread(target=process_tasks_from_queue, args=(task_queue,))
thread.start()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Example of adding a new task on mouse click
            task_priority = random.randint(1, 10)
            task_complexity = random.uniform(0.5, 1.5)
            new_task = Task("User Task", task_priority, task_complexity)
            add_tasks_to_queue(task_queue, new_task)

    # Clear the screen
    screen.fill(WHITE)

    # Draw tasks (this would be more complex in a real implementation)
    list_tasks = list(task_queue.queue)
    for i, task in enumerate(list_tasks):
        pygame.draw.rect(screen, GREEN if i % 2 == 0 else RED, (50, 50 + 30 * i, 200, 20))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

# Clean up
pygame.quit()
sys.exit()
