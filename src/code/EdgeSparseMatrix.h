#ifndef EDGE_SPARSE_MATRIX_H
#define EDGE_SPARSE_MATRIX_H

#include <unordered_map>
#include <functional>
#include <vector>

// Hash function for std::pair<int,int> - for undirected edges
namespace std {
    template<>
    struct hash<pair<int, int>> {
        inline size_t operator()(const pair<int, int>& p) const noexcept {
            // Faster hash combining using Cantor pairing or bit manipulation
            // This avoids multiplication in the inner loop
            return static_cast<size_t>(p.first) ^ 
                  (static_cast<size_t>(p.second) << (sizeof(int) * 4));
        }
    };
}

// Equality comparison for unordered_map
struct PairEqual {
    inline bool operator()(const std::pair<int, int>& a, 
                          const std::pair<int, int>& b) const noexcept {
        return a.first == b.first && a.second == b.second;
    }
};

// Template class for a sparse matrix using hashmap directly with paired indices
template<typename T>
class SparseMatrix {
private:
    std::unordered_map<std::pair<int, int>, T, std::hash<std::pair<int, int>>, PairEqual> data;
    T default_value;
    
    // Utility function to normalize an edge (i,j) -> (min(i,j), max(i,j))
    inline void normalize(int& i, int& j) const noexcept {
        if (i > j) std::swap(i, j);
    }

public:
    // Constructor with capacity hint
    SparseMatrix(T default_val = T(), size_t initial_capacity = 0) : default_value(default_val) {
        if (initial_capacity > 0) {
            data.reserve(initial_capacity);
        }
    }

    // Fast set method
    inline void set(int i, int j, const T& value) {
        normalize(i, j);
        data[std::make_pair(i, j)] = value;
    }

    // Fast set method with move semantics
    inline void set(int i, int j, T&& value) {
        normalize(i, j);
        data[std::make_pair(i, j)] = std::move(value);
    }

    // Set with pre-normalized indices (when you know i <= j)
    inline void set_normalized(int i, int j, const T& value) {
        data[std::make_pair(i, j)] = value;
    }

    // Fast get method - minimizes temporary object creation
    inline T get(int i, int j) const {
        normalize(i, j);
        auto it = data.find(std::make_pair(i, j));
        if (it != data.end()) {
            return it->second;
        }
        return default_value;
    }

    // Get reference to the value (for modification)
    inline T& operator()(int i, int j) {
        normalize(i, j);
        return data[std::make_pair(i, j)];
    }

    // Set group method - optimized to normalize only once per vertex
    inline void set_group(int i, const std::vector<int>& j_vertices, const T& value) {
        for (int j : j_vertices) {
            int i_copy = i, j_copy = j;
            normalize(i_copy, j_copy);
            data[std::make_pair(i_copy, j_copy)] = value;
        }
    }

    // Batch get method to improve locality
    inline std::vector<T> get_group(int i, const std::vector<int>& j_vertices) const {
        std::vector<T> results;
        results.reserve(j_vertices.size());
        for (int j : j_vertices) {
            results.push_back(get(i, j));
        }
        return results;
    }

    // Direct access to the map for advanced users
    inline auto& get_data() {
        return data;
    }

    // Set default value
    inline void setDefault(const T& value) {
        default_value = value;
    }

    // Check existence - optimized to avoid temporary pair creation where possible
    inline bool exists(int i, int j) const {
        normalize(i, j);
        return data.find(std::make_pair(i, j)) != data.end();
    }

    // Clear the matrix
    inline void clear() {
        data.clear();
    }

    // Get size
    inline size_t size() const {
        return data.size();
    }
    
    // Calculate load factor
    inline float load_factor() const {
        return data.load_factor();
    }
    
    // Rehash to optimize memory usage
    inline void rehash(size_t count) {
        data.rehash(count);
    }
};

#endif // EDGE_SPARSE_MATRIX_H
