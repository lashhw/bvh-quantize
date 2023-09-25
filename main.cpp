#include "build.hpp"
#include "traverse.hpp"

int main(int argc, char *argv[]) {
    arg_t arg = parse_arg(argc, argv);

    std::cout << "loading..." << std::endl;
    std::vector<trig_t> trigs = load_trigs(arg.model_file);

    std::cout << "building..." << std::endl;
    bvh_t bvh = build_bvh(trigs);

    std::cout << "clustering..." << std::endl;
    int_bvh_t int_bvh = build_int_bvh(arg.t_trv_int, arg.t_switch, arg.t_ist, trigs, bvh);

    std::cout << "visualizing..." << std::endl;
    gen_tree_visualization(int_bvh);
    gen_scene_visualization(bvh, int_bvh);
}