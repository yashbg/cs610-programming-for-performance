#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <semaphore.h>
#include <cerrno>

int n;
sem_t full;
sem_t empty;

std::multiset<std::string> x;
std::queue<std::string> y;
std::map<std::string, int> z;
pthread_mutex_t x_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t y_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t z_mutex = PTHREAD_MUTEX_INITIALIZER;

void *producer(void *arg) {
    int errcode = pthread_mutex_lock(&x_mutex);
    if (errcode) {
        throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
    }
    
    std::string file = *x.begin();
    x.erase(x.begin());

    errcode = pthread_mutex_unlock(&x_mutex);
    if (errcode) {
        throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
    }

    std::ifstream ifs(file);
    std::string line;

    while (getline(ifs, line)) {
        if (sem_wait(&empty)) {
            throw std::runtime_error("Error waiting on semaphore" + std::string(std::strerror(errno)));
        }

        errcode = pthread_mutex_lock(&y_mutex);
        if (errcode) {
            throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
        }
        
        y.push(line);

        errcode = pthread_mutex_unlock(&y_mutex);
        if (errcode) {
            throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
        }

        if (sem_post(&full)) {
            throw std::runtime_error("Error posting on semaphore" + std::string(std::strerror(errno)));
        }
    }

    return NULL;
}

void *consumer(void *arg) {
    while (true) {
        if (sem_wait(&full)) {
            throw std::runtime_error("Error waiting on semaphore" + std::string(std::strerror(errno)));
        }

        int errcode = pthread_mutex_lock(&y_mutex);
        if (errcode) {
            throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
        }
        
        std::string line = y.front();
        if (line == "") {
            // poison pill
            errcode = pthread_mutex_unlock(&y_mutex);
            if (errcode) {
                throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
            }

            if (sem_post(&full)) {
                throw std::runtime_error("Error posting on semaphore" + std::string(std::strerror(errno)));
            }

            return NULL;
        }

        y.pop();

        errcode = pthread_mutex_unlock(&y_mutex);
        if (errcode) {
            throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
        }

        if (sem_post(&empty)) {
            throw std::runtime_error("Error posting on semaphore" + std::string(std::strerror(errno)));
        }

        std::stringstream ss(line);
        std::string word;
        while (ss >> word) {
            errcode = pthread_mutex_lock(&z_mutex);
            if (errcode) {
                throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
            }

            std::string finalWord;
            for (int i = 0; i < word.size(); i++) {
                if (('A' <= word[i] && word[i] <= 'Z') || ('a' <= word[i] && word[i] <= 'z')) {
                    finalWord += word[i];
                }
            }
            
            z[finalWord]++;

            errcode = pthread_mutex_unlock(&z_mutex);
            if (errcode) {
                throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
            }
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string input(argv[1]);

    std::ifstream ifs(input);
    std::string file;
    while (ifs >> file) {
        x.insert(file);
    }

    n = x.size();

    if (sem_init(&full, 0, 0)) {
        throw std::runtime_error("Error initializing semaphore" + std::string(std::strerror(errno)));
    }
    
    if (sem_init(&empty, 0, n)) {
        throw std::runtime_error("Error initializing semaphore" + std::string(std::strerror(errno)));
    }

    pthread_t producerThreads[n];
    for (int i = 0; i < n; i++) {
        int errcode = pthread_create(&producerThreads[i], NULL, producer, NULL);
        if (errcode) {
            throw std::runtime_error("Error creating thread, errcode = " + std::to_string(errcode));
        }
    }

    int m = n;
    pthread_t consumerThreads[m];
    for (int i = 0; i < m; i++) {
        int errcode = pthread_create(&consumerThreads[i], NULL, consumer, NULL);
        if (errcode) {
            throw std::runtime_error("Error creating thread, errcode = " + std::to_string(errcode));
        }
    }

    for (int i = 0; i < n; i++) {
        int errcode = pthread_join(producerThreads[i], NULL);
        if (errcode) {
            throw std::runtime_error("Error joining thread, errcode = " + std::to_string(errcode));
        }
    }

    // push poison pill
    if (sem_wait(&empty)) {
        throw std::runtime_error("Error waiting on semaphore" + std::string(std::strerror(errno)));
    }

    int errcode = pthread_mutex_lock(&y_mutex);
    if (errcode) {
        throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
    }

    y.push("");

    errcode = pthread_mutex_unlock(&y_mutex);
    if (errcode) {
        throw std::runtime_error("Error unlocking mutex, errcode = " + std::to_string(errcode));
    }

    if (sem_post(&full)) {
        throw std::runtime_error("Error posting on semaphore" + std::string(std::strerror(errno)));
    }

    for (int i = 0; i < m; i++) {
        int errcode = pthread_join(consumerThreads[i], NULL);
        if (errcode) {
            throw std::runtime_error("Error joining thread, errcode = " + std::to_string(errcode));
        }
    }

    for (auto &word : z) {
        std::cout << word.first << " " << word.second << std::endl;
    }

    return 0;
}
