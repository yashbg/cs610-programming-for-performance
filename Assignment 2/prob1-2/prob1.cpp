#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>
#include <set>

std::multiset<std::string> x;

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string input(argv[1]);

    std::ifstream ifs(input);
    std::string line;
    while (ifs >> line) {
        x.insert(line);
    }

    int n = x.size();

    return 0;
}
