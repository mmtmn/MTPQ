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
const sf::Color AGENT_COLOR = sf::Color(0, 0, 255);
const sf::Color FOOD_COLOR = sf::Color(0, 255, 0);
constexpr int AGENT_SIZE = 10;
constexpr int FOOD_SIZE = 5;
constexpr int MATURITY_TIME = 10; // in seconds
constexpr int MAX_AGE = 180; // in seconds
constexpr int NUM_FOOD_ITEMS = 1000; // Number of food items to add

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
        // cout << "Drawing food at position: " << position.x << ", " << position.y << endl; // Comment out for less verbosity
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

    Agent(string n, float x, float y, float e) : name(std::move(n)), position(x, y), direction(1, 0), energy(e) {
        birthTime = steady_clock::now();
        lastReproduceTime = birthTime;
        cout << "Agent " << name << " created at position (" << x << ", " << y << ") with energy " << e << endl; // Debug statement
    }


    Agent(Agent&& other) noexcept 
        : name(std::move(other.name)), position(other.position), direction(other.direction), tasks(std::move(other.tasks)), 
          energy(other.energy.load()), alive(other.alive), stress(other.stress), birthTime(other.birthTime), lastReproduceTime(other.lastReproduceTime) {}

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

    void draw(RenderWindow& window) {
        CircleShape shape(AGENT_SIZE);
        shape.setFillColor(AGENT_COLOR);
        shape.setPosition(position.x, position.y);
        window.draw(shape);
        // cout << "Drawing agent " << name << " at position: " << position.x << ", " << position.y << endl; // Comment out for less verbosity
    }

    void moveAgent() {
        cout << "moveAgent called" << endl; // Debug statement
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(-1, 1);
        direction = sf::Vector2f(dist(gen), dist(gen));
        position += direction;
        position = sf::Vector2f(max(0.f, min(position.x, float(SCREEN_WIDTH))),
                            max(0.f, min(position.y, float(SCREEN_HEIGHT))));
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

            string childName = name + "_child";
            float childX = position.x;
            float childY = position.y;
            float childEnergy = 50;

            cout << "Creating new agent with name: " << childName << ", position: (" << childX << ", " << childY << "), energy: " << childEnergy << endl;

            try {
                agents.push_back(Agent(childName, childX, childY, childEnergy));
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

    void senseEnvironment(const vector<Food>& foods, const vector<Agent>& agents) {
        vision(foods);
        hearing(agents);
        touch(agents);
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

    void touch(const vector<Agent>& agents) {
        cout << "touch called" << endl; // Debug statement
        for (const auto& agent : agents) {
            if (this != &agent) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < AGENT_SIZE) {
                    cout << name << " touches " << agent.name << endl;
                    stress += 10;
                }
            }
        }
    }

    void decideNextAction(vector<Agent>& agents, network<sequential>& net) {
        cout << "decideNextAction called" << endl; // Debug statement
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.1, 0.5);

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

        if (age >= MAX_AGE) {
            alive = false;
            cout << name << " has reached maximum age and died." << endl;
        }
    }

    void update(const vector<Food>& foods, vector<Agent>& agents, network<sequential>& net) {
        if (alive) {
            cout << "Updating agent: " << name << endl;
            senseEnvironment(foods, agents);
            decideNextAction(agents, net);
            cout << "Processing tasks for agent: " << name << endl;
            processTasks();
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
    agents.emplace_back("Agent1", 400, 300, 100);

    // Generate random food items
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disX(0, SCREEN_WIDTH);
    uniform_int_distribution<> disY(0, SCREEN_HEIGHT);
    for (int i = 0; i < NUM_FOOD_ITEMS; ++i) {
        foods.emplace_back(disX(gen), disY(gen));
    }

    // Comment out threading for now
    // for (auto& agent : agents) {
    //     agent.simulateLife();
    // }
}

void runSimulation() {
    RenderWindow window(VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Agent-Based Simulation");
    window.setFramerateLimit(60);

    initializeSimulation();
    cout << "Simulation initialized" << endl;

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
            for (auto& agent : agents) {
                cout << "Updating and drawing agent: " << agent.name << endl; // Debug statement
                agent.update(foods, agents, net);
                agent.draw(window);
            }
        }

        window.display();
        cout << "Window updated and displayed" << endl;
    }
}

int main() {
    cout << "Starting simulation" << endl;
    runSimulation();
    return 0;
}
