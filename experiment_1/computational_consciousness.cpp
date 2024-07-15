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
#include <future>
#include <map>

using namespace std;
using namespace sf;
using namespace Eigen;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace std::chrono;

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;
const sf::Color BLUE_AGENT_COLOR = sf::Color(0, 0, 255);
const sf::Color RED_AGENT_COLOR = sf::Color(255, 0, 0);
const sf::Color GREEN_AGENT_COLOR = sf::Color(0, 255, 0);
const sf::Color FOOD_COLOR = sf::Color(112, 112, 112);
const sf::Color BACKGROUND_COLOR = sf::Color::White;
constexpr int AGENT_SIZE = 10;
constexpr int FOOD_SIZE = 5;
constexpr int VISION_RANGE = 50;
constexpr int HEARING_RANGE = 100;
constexpr int MATURITY_TIME = 18; // in seconds
constexpr int MAX_AGE = 100; // in seconds
constexpr int INITIAL_FOOD_ITEMS = 10000; // Initial number of food items
constexpr int MAX_FOOD_ITEMS = 2000; // Maximum number of food items in the environment

mutex mtx;
condition_variable cv;
priority_queue<pair<int, function<void()>>> taskQueue;
mutex agentsMutex;

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

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>;

private:
    vector<thread> workers;
    queue<function<void()>> tasks;

    mutex queue_mutex;
    condition_variable condition;
    atomic<bool> stop;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for(;;) {
                function<void()> task;
                {
                    unique_lock<mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop.load() || !this->tasks.empty(); });
                    if(this->stop.load() && this->tasks.empty())
                        return;
                    task = move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
}

template<class F>
auto ThreadPool::enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
    using return_type = typename std::result_of<F()>::type;
    auto task = make_shared<packaged_task<return_type()>>(forward<F>(f));
    future<return_type> res = task->get_future();
    {
        unique_lock<mutex> lock(queue_mutex);
        if(stop.load())
            throw runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for(thread &worker: workers)
        worker.join();
}

enum AgentType { BLUE, RED, GREEN };

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


class Agent {
public:
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
    int senseImportance[3] = {5, 5, 5}; // Importance of vision, hearing, and touch
    ThreadPool* threadPool;
    AgentType agentType;
    sf::Color agentColor;

    Agent(string n, float x, float y, float e, float s, const network<sequential>& parentBrain, AgentType type, ThreadPool* tp)
        : name(move(n)), position(x, y), direction(1, 0), energy(e), speed(s), brain(parentBrain), agentType(type), threadPool(tp) {
        birthTime = steady_clock::now();
        lastReproduceTime = birthTime;
        switch (agentType) {
            case BLUE: agentColor = BLUE_AGENT_COLOR; break;
            case RED: agentColor = RED_AGENT_COLOR; break;
            case GREEN: agentColor = GREEN_AGENT_COLOR; break;
        }
        cout << "Agent " << name << " created at position (" << x << ", " << y << ") with energy " << e << " and speed " << s << endl;
    }

    Agent(Agent&& other) noexcept 
        : name(move(other.name)), position(other.position), direction(other.direction), tasks(move(other.tasks)), 
          energy(other.energy.load()), alive(other.alive), stress(other.stress), birthTime(other.birthTime), lastReproduceTime(other.lastReproduceTime), 
          speed(other.speed), brain(move(other.brain)), threadPool(other.threadPool), agentType(other.agentType), agentColor(other.agentColor) {}

    Agent& operator=(Agent&& other) noexcept {
        if (this != &other) {
            name = move(other.name);
            position = other.position;
            direction = other.direction;
            tasks = move(other.tasks);
            energy.store(other.energy.load());
            alive = other.alive;
            stress = other.stress;
            birthTime = other.birthTime;
            lastReproduceTime = other.lastReproduceTime;
            speed = other.speed;
            brain = move(other.brain);
            threadPool = other.threadPool;
            agentType = other.agentType;
            agentColor = other.agentColor;
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
                } catch (const std::exception& e) {
                    cout << "Exception in task action: " << e.what() << endl;
                } catch (...) {
                    cout << "Unknown exception in task action" << endl;
                }
                float currentEnergy = energy.load();
                energy.store(currentEnergy - task.complexity);
            } else {
                cout << "Not enough energy for task: " << task.name << endl;
            }
        }
    }

    void simulateLife() {
        thread taskProcessor(&Agent::processTasks, this);
        taskProcessor.detach();
    }

    void draw(RenderWindow& window, bool showRanges) {
        CircleShape shape(AGENT_SIZE);
        shape.setFillColor(alive ? agentColor : sf::Color(100, 100, 100)); // Grey if dead
        shape.setPosition(position.x, position.y);
        window.draw(shape);

        if (showRanges) {
            CircleShape visionCircle(VISION_RANGE);
            visionCircle.setFillColor(sf::Color(0, 0, 255, 50)); // Transparent blue for vision
            visionCircle.setPosition(position.x - VISION_RANGE + AGENT_SIZE / 2, position.y - VISION_RANGE + AGENT_SIZE / 2);
            window.draw(visionCircle);

            CircleShape hearingCircle(HEARING_RANGE);
            hearingCircle.setFillColor(sf::Color(0, 255, 0, 50)); // Transparent green for hearing
            hearingCircle.setPosition(position.x - HEARING_RANGE + AGENT_SIZE / 2, position.y - HEARING_RANGE + AGENT_SIZE / 2);
            window.draw(hearingCircle);
        }
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
                    for (auto& weight : *weight_vec) {
                        normal_distribution<float> mutation(0.0, 0.1);
                        weight += mutation(gen);
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
                    agents.push_back(Agent(childName, childX, childY, childEnergy, childSpeed, childBrain, agentType, threadPool));
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

    void reward(float value) {
        energy.store(energy.load() + value);
        senseImportance[0] = min(10, senseImportance[0] + int(value / 10)); // Increase importance of vision
        senseImportance[1] = min(10, senseImportance[1] + int(value / 10)); // Increase importance of hearing
        senseImportance[2] = min(10, senseImportance[2] + int(value / 10)); // Increase importance of touch
    }

    void punish(float value) {
        energy.store(energy.load() - value);
        senseImportance[0] = max(0, senseImportance[0] - int(value / 10)); // Decrease importance of vision
        senseImportance[1] = max(0, senseImportance[1] - int(value / 10)); // Decrease importance of hearing
        senseImportance[2] = max(0, senseImportance[2] - int(value / 10)); // Decrease importance of touch
    }

    void touchFood(vector<Food>& foods) {
        cout << "touchFood called" << endl; // Debug statement
        for (auto it = foods.begin(); it != foods.end(); ++it) {
            float distance = sqrt((position.x - it->position.x) * (position.x - it->position.x) +
                                  (position.y - it->position.y) * (position.y - it->position.y));
            if (distance < AGENT_SIZE) {
                reward(10);
                cout << name << " touched food and increased energy to " << energy << endl;
                foods.erase(it); // Remove food once collected
            } else {
               ++it; // Move to the next food item
            }
        }
    }

    void hearing(const vector<Agent>& agents) {
        cout << "hearing called" << endl; // Debug statement
        float hearingRange = 100;
        for (const auto& agent : agents) {
            if (this != &agent) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < hearingRange) {
                    cout << name << " hears " << agent.name << endl;
                }
            }
        }
    }

    void touchAgents(vector<Agent>& agents) {
        cout << "touchAgents called" << endl; // Debug statement
        for (auto& agent : agents) {
            if (this != &agent) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < AGENT_SIZE * 2) { // Adjust for agent size
                    cout << name << " touches " << agent.name << endl;
                    stress += 10;
                    // Collision avoidance
                    direction = -direction;
                    position += direction * speed;

                    // Predation
                    if ((agentType == BLUE && agent.agentType == RED) ||
                        (agentType == RED && agent.agentType == GREEN) ||
                        (agentType == GREEN && agent.agentType == BLUE)) {
                        reward(agent.energy.load());
                        agent.alive = false;
                        cout << name << " has eaten " << agent.name << " and increased energy to " << energy << endl;
                    }
                }
            }
        }
    }

    void senseEnvironment(vector<Food>& foods, vector<Agent>& agents) {
        future<void> visionFuture = threadPool->enqueue([this, &foods]() { vision(foods); });
        future<void> hearingFuture = threadPool->enqueue([this, &agents]() { hearing(agents); });
        future<void> touchFuture = threadPool->enqueue([this, &agents]() { touchAgents(agents); });

        visionFuture.get();
        hearingFuture.get();
        touchFuture.get();
    }

    void vision(const vector<Food>& foods) {
        cout << "vision called" << endl; // Debug statement
        float visionRange = 50;
        float visionAngle = M_PI / 4;
        for (const auto& food : foods) {
            sf::Vector2f toFood = food.position - position;
            float distance = sqrt(toFood.x * toFood.x + toFood.y * toFood.y);
            if (distance < visionRange) {
                float angle = acos((direction.x * toFood.x + direction.y * toFood.y) /
                                   (sqrt(direction.x * direction.x + direction.y * direction.y) * distance));
                if (angle < visionAngle) {
                    float currentEnergy = energy.load();
                    energy.store(currentEnergy + 10);
                    cout << name << " found food by vision and increased energy to " << energy << endl;
                    break;
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

vector<Agent> agents;
vector<Food> foods;
network<sequential> net;
ThreadPool threadPool(4);
bool showRanges = false;

void initializeSimulation() {
    net = setupNeuralNetwork();
    agents.reserve(10000); // Preallocate space to avoid reallocation
    agents.emplace_back("Agent1", 400, 300, 100, 1.0f, net, BLUE, &threadPool);
    agents.emplace_back("Agent2", 200, 150, 100, 1.0f, net, RED, &threadPool);
    agents.emplace_back("Agent3", 600, 450, 100, 1.0f, net, GREEN, &threadPool);

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
    RenderWindow window(VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Multi Threaded Priority Queue Simulation");
    window.setFramerateLimit(60);

    initializeSimulation();
    cout << "Simulation initialized" << endl;

    thread foodRespawnThread(respawnFood);

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
            else if (event.type == Event::KeyPressed && event.key.code == Keyboard::Space) {
                showRanges = !showRanges; // Toggle range visibility
            }
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
                    it->draw(window, showRanges);
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
