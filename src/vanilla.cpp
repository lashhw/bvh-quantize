#include "build.hpp"
#include "traverse.hpp"
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include "unistd.h"

typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, trig_t> primitive_intersector_t;

// prevent gcc from optimizing out result
__attribute__((__used__)) std::optional<primitive_intersector_t::Result> full_result;

int main() {
    // perf control fifo
    char* ctl_fd_str = getenv("CTL_FD");
    assert(ctl_fd_str != nullptr);
    char* ack_fd_str = getenv("ACK_FD");
    assert(ack_fd_str != nullptr);
    int ctl_fd = std::stoi(ctl_fd_str);
    int ack_fd = std::stoi(ack_fd_str);
    char ack[5];

    arg_t arg = {
        .model_file = strdup("../data/scene/kitchen.ply"),
        .t_trv_int = -1,
        .t_switch = -1,
        .t_ist = -1,
        .ray_file = strdup("../data/scene/kitchen.ray")
    };

    std::cout << "loading scene..." << std::endl;
    std::vector<trig_t> trigs = load_trigs(arg.model_file);

    std::cout << "building..." << std::endl;
    bvh_t bvh = build_bvh(trigs);

    std::cout << "loading rays..." << std::endl;
    std::vector<ray_t> rays;
    std::ifstream ray_fs(arg.ray_file);
    for (float r[7]; ray_fs.read((char*)r, 7 * sizeof(float)); ) {
        rays.emplace_back(
            vector_t(r[0], r[1], r[2]),
            vector_t(r[3], r[4], r[5]),
            0.f,
            r[6]
        );
    }

    // start profiling
    std::cout << "traversing..." << std::endl;
    write(ctl_fd, "enable\n", 8);
    read(ack_fd, ack, 5);
    assert(strncmp(ack, "ack\n", 5) == 0);

    traverser_t full_traverser(bvh);
    primitive_intersector_t primitive_intersector(bvh, trigs.data());
    for (ray_t &ray : rays)
        full_result = full_traverser.traverse(ray, primitive_intersector);

    // finish profiling
    write(ctl_fd, "disable\n", 9);
    read(ack_fd, ack, 5);
    assert(strncmp(ack, "ack\n", 5) == 0);

    std::cout << rays.size() << " rays traversed." << std::endl;
}
