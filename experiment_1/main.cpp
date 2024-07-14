#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <tiny_dnn/tiny_dnn.h>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace sf;
using namespace Eigen;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace std::chrono;

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;
const sf::Color BACKGROUND_COLOR = sf::Color(255, 255, 255);
const sf::Color AGENT_COLOR_BLUE = sf::Color(0, 0, 255);
const sf::Color AGENT_COLOR_RED = sf::Color(255, 0, 0);
const sf::Color AGENT_COLOR_GREEN = sf::Color(0, 255, 0);
const sf::Color FOOD_COLOR = sf::Color(0, 255, 0);
constexpr int AGENT_SIZE = 10;
constexpr int FOOD_SIZE = 5;
constexpr int MATURITY_TIME = 10; // in seconds
constexpr int MAX_AGE = 100; // in seconds
constexpr int INITIAL_FOOD_ITEMS = 10000; // Initial number of food items
constexpr int MAX_FOOD_ITEMS = 2000; // Maximum number of food items in the environment

mutex mtx;
condition_variable cv;
priority_queue<pair<int, function<void()>>> taskQueue;
mutex agentsMutex;

enum AgentType { BLUE, RED, GREEN };

struct Food {
    sf::Vector2f position;

    Food(float x, float y) : position(x, y) {}

    void draw(RenderWindow& window) {
        CircleShape shape(FOOD_SIZE);
        shape.setFillColor(FOOD_COLOR);
        shape.setPosition(position.x, position.y);
        window.draw(shape);
    }
};

struct Task {
    string name;
    int priority;
    double complexity;
    function<void()> action;

    Task(string n, int p, double c, function<void()> a) : name(move(n)), priority(p), complexity(c), action(move(a)) {}

    bool operator<(const Task& other) const {
        return priority > other.priority; // Higher priority first
    }
};

Vector4f getRealTimeData();
float predictPriority(network<sequential>& net, const Vector4f& data);

struct Agent {
    string name;
    sf::Vector2f position;
    sf::Vector2f direction;
    priority_queue<Task> tasks;
    atomic<float> energy;
    bool alive = true;
    float stress = 0;
    time_point<steady_clock> birthTime;
    time_point<steady_clock> lastReproduceTime;
    float speed; // Genetic trait
    network<sequential> brain; // Neural network for decision making
    AgentType type; // Type of agent (BLUE, RED, GREEN)

    Agent(string n, float x, float y, float e, float s, const network<sequential>& parentBrain, AgentType t)
        : name(std::move(n)), position(x, y), direction(1, 0), energy(e), speed(s), brain(parentBrain), type(t) {
        birthTime = steady_clock::now();
        lastReproduceTime = birthTime;
        cout << "Agent " << name << " created at position (" << x << ", " << y << ") with energy " << e << " and speed " << s << endl;
    }

    Agent(Agent&& other) noexcept 
        : name(std::move(other.name)), position(other.position), direction(other.direction), tasks(std::move(other.tasks)), 
          energy(other.energy.load()), alive(other.alive), stress(other.stress), birthTime(other.birthTime), lastReproduceTime(other.lastReproduceTime), speed(other.speed), brain(std::move(other.brain)), type(other.type) {}

    Agent& operator=(Agent&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            position = other.position;
            direction = other.direction;
            tasks = std::move(other.tasks);
            energy.store(other.energy.load());
            alive = other.alive;
            stress = other.stress;
            birthTime = other.birthTime;
            lastReproduceTime = other.lastReproduceTime;
            speed = other.speed;
            brain = std::move(other.brain);
            type = other.type;
        }
        return *this;
    }

    void addTask(const Task& task) {
        lock_guard<mutex> lock(mtx);
        tasks.push(task);
        cv.notify_all();
        cout << "Task added: " << task.name << " with priority " << task.priority << endl;
    }

    void processTasks() {
        while (alive) {
            unique_lock<mutex> lock(mtx);
            if (tasks.empty()) {
                cout << "No tasks to process for agent: " << name << endl;
                return;
            }

            Task task = tasks.top();
            tasks.pop();
            cout << "Popped task: " << task.name << " with priority " << task.priority << endl;
            lock.unlock();
            if (energy >= task.complexity) {
                cout << "Executing task: " << task.name << " with priority " << task.priority << endl;
                try {
                    task.action();
                    // Provide a small energy boost upon successful task completion
                    energy.store(energy.load() + 1.0f);
                } catch (const std::exception& e) {
                    cout << "Exception in task action: " << e.what() << endl;
                } catch (...) {
                    cout << "Unknown exception in task action" << endl;
                }
                float currentEnergy = energy.load();
                energy.store(currentEnergy - task.complexity);
            } else {
                cout << "Not enough energy for task: " << task.name << endl;
                alive = false; // Mark as dead due to energy depletion
            }
        }
    }

    void simulateLife() {
        thread taskProcessor(&Agent::processTasks, this);
        taskProcessor.detach();
    }

    void draw(RenderWindow& window) {
        CircleShape shape(AGENT_SIZE);
        if (type == BLUE) {
            shape.setFillColor(alive ? AGENT_COLOR_BLUE : sf::Color(100, 100, 100)); // Grey if dead
        } else if (type == RED) {
            shape.setFillColor(alive ? AGENT_COLOR_RED : sf::Color(100, 100, 100)); // Grey if dead
        } else if (type == GREEN) {
            shape.setFillColor(alive ? AGENT_COLOR_GREEN : sf::Color(100, 100, 100)); // Grey if dead
        }
        shape.setPosition(position.x, position.y);
        window.draw(shape);
    }

    void moveAgent() {
        cout << "moveAgent called" << endl; // Debug statement
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(-1, 1);
        direction = sf::Vector2f(dist(gen), dist(gen));
        position += direction * speed;

        // Boundary conditions
        position.x = max(0.f, min(position.x, float(SCREEN_WIDTH - AGENT_SIZE)));
        position.y = max(0.f, min(position.y, float(SCREEN_HEIGHT - AGENT_SIZE)));
        
        cout << name << " moved to " << position.x << ", " << position.y << endl;
    }

    void stayPut() {
        cout << "stayPut called" << endl; // Debug statement
        cout << name << " is staying put." << endl;
    }

    void reproduce(vector<Agent>& agents) {
        cout << "reproduce called" << endl; // Debug statement
        auto now = steady_clock::now();
        auto age = chrono::duration_cast<chrono::seconds>(now - birthTime).count();
        auto timeSinceLastReproduce = chrono::duration_cast<chrono::seconds>(now - lastReproduceTime).count();

        cout << "Agent age: " << age << " seconds" << endl;
        cout << "Time since last reproduce: " << timeSinceLastReproduce << " seconds" << endl;
        cout << "Agent energy: " << energy.load() << endl;

        if (age >= MATURITY_TIME && timeSinceLastReproduce >= MATURITY_TIME && energy > 50) {
            cout << "Conditions met for reproduction" << endl;

            // Genetic algorithm: inherit speed with slight mutation
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> mutationDist(-0.1, 0.1);
            float childSpeed = speed + mutationDist(gen);

            // Mutate the neural network slightly
            network<sequential> childBrain = brain;
            for (auto& layer : childBrain) {
                for (auto& weight_vec : layer->weights()) {
                    for (size_t i = 0; i < weight_vec->size(); ++i) {
                        if (i % 2 == 0) { // Mutate 50% of the weights
                            normal_distribution<float> mutation(0.0, 0.1);
                            (*weight_vec)[i] += mutation(gen);
                        }
                    }
                }
            }

            string childName = name + "_child";
            float childX, childY;
            bool spaceFound = false;
            uniform_int_distribution<> offsetDist(-2 * AGENT_SIZE, 2 * AGENT_SIZE);

            // Find a suitable position for the child
            for (int attempts = 0; attempts < 10; ++attempts) {
                float offsetX = offsetDist(gen);
                float offsetY = offsetDist(gen);
                childX = position.x + offsetX;
                childY = position.y + offsetY;

                // Ensure the child is within boundaries
                childX = max(0.f, min(childX, float(SCREEN_WIDTH - AGENT_SIZE)));
                childY = max(0.f, min(childY, float(SCREEN_HEIGHT - AGENT_SIZE)));

                // Check if the space is occupied by another agent
                bool overlap = false;
                for (const auto& agent : agents) {
                    float distance = sqrt((childX - agent.position.x) * (childX - agent.position.x) +
                                          (childY - agent.position.y) * (childY - agent.position.y));
                    if (distance < AGENT_SIZE * 2) {
                        overlap = true;
                        break;
                    }
                }

                if (!overlap) {
                    spaceFound = true;
                    break;
                }
            }

            if (spaceFound) {
                float childEnergy = 50;
                cout << "Creating new agent with name: " << childName << ", position: (" << childX << ", " << childY << "), energy: " << childEnergy << ", speed: " << childSpeed << endl;

                try {
                    agents.push_back(Agent(childName, childX, childY, childEnergy, childSpeed, childBrain, type));
                    cout << "New agent added" << endl;
                } catch (const std::exception& e) {
                    cout << "Exception in reproduce: " << e.what() << endl;
                } catch (...) {
                    cout << "Unknown exception in reproduce" << endl;
                }

                float currentEnergy = energy.load();
                energy.store(currentEnergy - 50);
                lastReproduceTime = now;
                cout << name << " has reproduced." << endl;
            } else {
                cout << "No suitable space found for new agent." << endl;
            }
        } else {
            if (age < MATURITY_TIME) {
                cout << "Agent not mature enough to reproduce." << endl;
            }
            if (timeSinceLastReproduce < MATURITY_TIME) {
                cout << "Not enough time since last reproduce." << endl;
            }
            if (energy <= 50) {
                cout << "Not enough energy to reproduce." << endl;
            }
        }
    }

    void senseEnvironment(vector<Food>& foods, vector<Agent>& agents) {
        touchFood(foods);
        touchAgents(agents);
    }

    void touchFood(vector<Food>& foods) {
        cout << "touchFood called" << endl; // Debug statement
        for (auto it = foods.begin(); it != foods.end(); ++it) {
            float distance = sqrt((position.x - it->position.x) * (position.x - it->position.x) +
                                  (position.y - it->position.y) * (position.y - it->position.y));
            if (distance < AGENT_SIZE) {
                float currentEnergy = energy.load();
                energy.store(currentEnergy + 10);
                cout << name << " touched food and increased energy to " << energy << endl;
                foods.erase(it); // Remove food once collected
                break;
            }
        }
    }

    void touchAgents(vector<Agent>& agents) {
        cout << "touchAgents called" << endl; // Debug statement
        for (auto& agent : agents) {
            if (this != &agent && agent.alive) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < AGENT_SIZE * 2) { // Adjust for agent size
                    cout << name << " touches " << agent.name << endl;
                    stress += 10;
                    // Collision avoidance
                    direction = -direction;
                    position += direction * speed;

                    // Eating logic
                    if ((type == BLUE && agent.type == RED) ||
                        (type == RED && agent.type == GREEN) ||
                        (type == GREEN && agent.type == BLUE)) {
                        cout << name << " eats " << agent.name << endl;
                        energy.store(energy.load() + agent.energy.load());
                        agent.alive = false;
                    }
                }
            }
        }
    }

    void decideNextAction(vector<Agent>& agents, network<sequential>& net) {
        cout << "decideNextAction called" << endl; // Debug statement
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.1, 0.5); // Lowered complexity cost

        vector<function<void()>> actions = {bind(&Agent::moveAgent, this), bind(&Agent::stayPut, this)};
        auto now = steady_clock::now();
        auto age = chrono::duration_cast<chrono::seconds>(now - birthTime).count();

        if (age >= MATURITY_TIME && energy > 50) {
            actions.push_back(bind(&Agent::reproduce, this, ref(agents)));
        }

        for (const auto& action : actions) {
            float priority = predictPriority(net, getRealTimeData()) - float(stress);
            addTask(Task("Action", priority, dis(gen), action));
        }

        if (age >= MAX_AGE || energy <= 0) {
            alive = false;
            cout << name << " has died due to energy depletion or old age." << endl;
        }
    }

    void update(vector<Food>& foods, vector<Agent>& agents, network<sequential>& net) {
        if (alive) {
            cout << "Updating agent: " << name << endl;
            senseEnvironment(foods, agents);
            decideNextAction(agents, net);
            cout << "Processing tasks for agent: " << name << endl;
            processTasks();
        } else {
            cout << "Agent " << name << " is dead and will be removed." << endl;
        }
    }
};

network<sequential> setupNeuralNetwork() {
    network<sequential> net;
    net << fully_connected_layer(4, 20) << tanh_layer()
        << fully_connected_layer(20, 1) << sigmoid_layer();
    return net;
}

Vector4f getRealTimeData() {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d_light(500, 50), d_sound(300, 50), d_touch(150, 30);
    uniform_real_distribution<> stress(0, 1);
    Vector4f data(d_light(gen), d_sound(gen), d_touch(gen), stress(gen));
    return data;
}

float predictPriority(network<sequential>& net, const Vector4f& data) {
    vec_t input = {data[0], data[1], data[2], data[3]};
    return net.predict(input)[0];
}

vector<Agent> agents;
vector<Food> foods;
network<sequential> net;

void initializeSimulation() {
    net = setupNeuralNetwork();
    agents.reserve(10000); // Preallocate space to avoid reallocation
    agents.emplace_back("BlueAgent1", 100, 100, 100, 1.0f, net, BLUE);
    agents.emplace_back("RedAgent1", 200, 200, 100, 1.0f, net, RED);
    agents.emplace_back("GreenAgent1", 300, 300, 100, 1.0f, net, GREEN);

    // Generate initial food items
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disX(0, SCREEN_WIDTH);
    uniform_int_distribution<> disY(0, SCREEN_HEIGHT);
    for (int i = 0; i < INITIAL_FOOD_ITEMS; ++i) {
        foods.emplace_back(disX(gen), disY(gen));
    }
}

void respawnFood() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disX(0, SCREEN_WIDTH);
    uniform_int_distribution<> disY(0, SCREEN_HEIGHT);

    while (true) {
        this_thread::sleep_for(chrono::seconds(1));
        lock_guard<mutex> lock(mtx);
        if (foods.size() < MAX_FOOD_ITEMS) {
            foods.emplace_back(disX(gen), disY(gen));
            cout << "Respawned food at new location" << endl;
        }
    }
}

void runSimulation() {
    RenderWindow window(VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Agent-Based Simulation");
    window.setFramerateLimit(60);

    initializeSimulation();
    cout << "Simulation initialized" << endl;

    thread foodRespawnThread(respawnFood);

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(BACKGROUND_COLOR);

        for (auto& food : foods) {
            food.draw(window);
        }

        {
            lock_guard<mutex> lock(agentsMutex); // Lock the agents vector during the draw call
            auto it = agents.begin();
            while (it != agents.end()) {
                if (!it->alive) {
                    it = agents.erase(it); // Remove dead agents
                } else {
                    cout << "Updating and drawing agent: " << it->name << endl; // Debug statement
                    it->update(foods, agents, net);
                    it->draw(window);
                    ++it;
                }
            }
        }

        window.display();
        cout << "Window updated and displayed" << endl;
    }

    foodRespawnThread.join();
}

int main() {
    cout << "Starting simulation" << endl;
    runSimulation();
    return 0;
}
