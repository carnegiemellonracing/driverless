#include "quadTree.hpp"
#include <glm/common.hpp>

static constexpr node_id empty = node_id(-1);

Box::Box(glm::fvec2 _ul, glm::fvec2 _lr) : ul{_ul}, lr{_lr} {}

glm::fvec2 QuadNode::find_center(){
    return (bbox.ul + bbox.lr)/2;
}

void QuadTree::insert(glm::fvec2 cone) {
    QuadNode* cur = root;
    while() {
        int idx = 0;
        glm::fvec2 center = cur.find_center();
        
        if(cone.x < center.x) idx += 1;
        if(cone.y >= center.y) idx += 2;
        // 0 = UR, 1 = UL, 2 = LR, 3 = LL
        
        if(cur.leaf == true){
            QuadNode split = QuadNode();
            glm::fvec2 ul, br;
            switch(idx):{
                case 0:
                    ul = glm::fvec2(center.x, cur.bbox.ul.y);
                    lr = glm::fvec2(cur.bbox.lr.x, center.y);
                    break;
                case 1:
                    ul = glm::fvec2(cur.bbox.ul.x, cur.bbox.ul.y);
                    lr = glm::fvec2(center.x, center.y);
                    break;
                case 2:
                    ul = glm::fvec2(center.x, center.y);
                    lr = glm::fvec2(cur.bbox.lr.x, cur.bbox.lr.y);
                    break;
                case 3:
                    ul = glm::fvec2(cur.bbox.ul.x, center.y);
                    lr = glm::fvec2(center.x, cur.bbox.lr.y);
                    break;
                default:
                    break;
            }
                  
            split.bbox = Box(ul, lr);
            break;
        }else if(){
            cur = cur.nodeList[idx];
        }

    }

}

bool QuadNode::is_full(){
    for(auto i : nodeList)
        if(i==NULL) return false;
    return true;
}


bool QuadNode::is_full(index i){
    if(nodeList[i] == NULL) return false;
    return true;
}

bool Quad
