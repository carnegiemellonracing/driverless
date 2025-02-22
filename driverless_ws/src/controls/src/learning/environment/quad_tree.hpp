#include <vector>
#include <iostream>
#include <cstdint>
#include <glm/common.hpp>

struct Box {
    glm::fvec2 ul;
    glm::fvec2 lr;
    Box(glm::fvec2 _ul, glm::fvec2 _lr);
}

union NodeData {
    glm::fvec2 loc;
    Box bbox; // 0 = UR, 1 = UL, 2 = LR, 3 = LL
};

struct QuadNode{
    bool leaf;
    union NodeData data;
    std::array<std::unique_ptr<QuadNode>, 4> nodeList; // all nullptr if leaf

    public:
        bool is_full();
        glm::fvec2 find_center();
};

class QuadTree {
    public:
        void insert(glm::fvec2 cone);
        void build_track(vector<glm::fvec2> cones);
        vector<glm::fvec2> find_nearest_cones(glm::fvec point);
    private:
        QuadNode root;
}