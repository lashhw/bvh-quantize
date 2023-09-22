#include "build.hpp"

int main(int argc, char *argv[]) {
    arg_t arg = parse_arg(argc, argv);

    std::cout << "loading..." << std::endl;
    std::vector<triangle_t> triangles = load_triangles(arg.model_file);

    std::cout << "building..." << std::endl;
    bvh_t bvh = build_bvh(triangles);

    std::cout << "clustering..." << std::endl;
    int_bvh_t int_bvh = build_int_bvh(arg.t_trv_int, arg.t_switch, arg.t_ist, bvh);
}