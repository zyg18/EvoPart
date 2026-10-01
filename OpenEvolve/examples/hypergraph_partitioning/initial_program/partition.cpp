// Hypergraph partitioner: the whole program is in this one file.
//
// Usage: ./our_hparter <input.hgr> --k=K --eps=EPS --seed=S --out=<partition file>
//
// Input, hMETIS .hgr format (unweighted): the first non-comment line is
// "<num_hyperedges> <num_vertices>", then one line per hyperedge listing its
// 1-indexed vertex ids. Lines starting with '%' are comments.
//
// Output: one line per vertex, in vertex order, holding its part id in [0, k).
// Every part must hold at most floor((1 + eps) * ceil(n / k)) vertices.
//
// This starting version does no optimisation at all: vertex i goes to part
// i mod k, which is balanced but cuts almost every hyperedge.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static bool readHgr(const std::string &path, int &n, std::vector<std::vector<int>> &nets)
{
    std::ifstream in(path.c_str());
    if (!in.is_open()) return false;
    std::string line;
    int m = 0;
    n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream hs(line);
        if (!(hs >> m >> n)) return false;
        break;
    }
    if (m <= 0 || n <= 0) return false;
    nets.reserve(m);
    while ((int)nets.size() < m && std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ls(line);
        std::vector<int> pins;
        int v;
        while (ls >> v) {
            if (v < 1 || v > n) return false;
            pins.push_back(v - 1);
        }
        nets.push_back(pins);
    }
    return (int)nets.size() == m;
}

int main(int argc, char **argv)
{
    std::string input, out;
    int k = 2;
    double eps = 0.02;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--k=", 0) == 0) k = atoi(a.c_str() + 4);
        else if (a.rfind("--eps=", 0) == 0) eps = atof(a.c_str() + 6);
        else if (a.rfind("--seed=", 0) == 0) seed = (unsigned)strtoul(a.c_str() + 7, nullptr, 10);
        else if (a.rfind("--out=", 0) == 0) out = a.substr(6);
        else if (a[0] != '-') input = a;
    }
    if (input.empty() || out.empty() || k < 2) {
        std::cerr << "usage: " << argv[0] << " <input.hgr> --k=K --eps=EPS --seed=S --out=FILE\n";
        return 2;
    }
    (void)seed;

    int n;
    std::vector<std::vector<int>> nets;
    if (!readHgr(input, n, nets)) {
        std::cerr << "error: cannot read " << input << "\n";
        return 1;
    }

    std::vector<int> part(n);
    for (int v = 0; v < n; ++v) part[v] = v % k;

    long long cut = 0;
    for (const auto &pins : nets) {
        for (size_t j = 1; j < pins.size(); ++j)
            if (part[pins[j]] != part[pins[0]]) { ++cut; break; }
    }
    long long cap = (long long)((1.0 + eps) * ((n + k - 1) / k));
    std::vector<int> sizes(k, 0);
    for (int v = 0; v < n; ++v) ++sizes[part[v]];
    bool feasible = true;
    for (int p = 0; p < k; ++p) if (sizes[p] > cap) feasible = false;

    std::ofstream of(out.c_str());
    if (!of.is_open()) {
        std::cerr << "error: cannot write " << out << "\n";
        return 1;
    }
    for (int v = 0; v < n; ++v) of << part[v] << "\n";
    of.close();

    std::cout << "RESULT cut=" << cut << " feasible=" << (feasible ? 1 : 0) << std::endl;
    return 0;
}
