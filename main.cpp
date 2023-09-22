#include "build.hpp"

int main(int argc, char *argv[]) {
    arg_t arg = parse_arg(argc, argv);

    std::cout << "loading..." << std::endl;
    std::vector<triangle_t> triangles = load_triangles(arg.model_file);

    std::cout << "building..." << std::endl;
    bvh_t bvh = build_bvh(triangles);

    std::cout << "clustering..." << std::endl;
    std::vector<policy_t> policy = get_policy(arg.t_trv_int, arg.t_switch, arg.t_ist, bvh);
    int_bvh_t int_bvh = build_int_bvh(bvh, policy);
}