#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string input(argv[1]);
    std::vector<std::string> lines;

    std::ifstream ifs(input);
    std::string line;
    while (ifs >> line) {
        lines.push_back(line);
    }

    return 0;
}
