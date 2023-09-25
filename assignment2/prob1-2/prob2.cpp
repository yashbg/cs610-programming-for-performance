#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <cassert>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include "concurrentqueue.h"

#define MAX_QUEUE_SIZE 10

std::multiset<std::string> x;
moodycamel::ConcurrentQueue<std::string> y(MAX_QUEUE_SIZE);
std::map<std::string, int> z;
pthread_mutex_t x_mutex = PTHREAD_MUTEX_INITIALIZER;
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
        while (!y.try_enqueue(line));
    }

    return NULL;
}

void *consumer(void *arg) {
    while (true) {
        std::string line;
        while (!y.try_dequeue(line));

        if (line == "") {
            // poison pill
            while (!y.try_enqueue(line));

            return NULL;
        }

        std::stringstream ss(line);
        std::string word;
        while (ss >> word) {
            int errcode = pthread_mutex_lock(&z_mutex);
            if (errcode) {
                throw std::runtime_error("Error locking mutex, errcode = " + std::to_string(errcode));
            }

            // remove punctuations
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

    int n = x.size();

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
    std::string poisonPill = "";
    while (!y.try_enqueue(poisonPill));

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
