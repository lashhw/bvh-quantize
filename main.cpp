#include <iostream>
#include <vector>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include "happly/happly.h"

constexpr size_t max_leaf_size = 6;
constexpr int quant_num = (1 << 8) - 1;
constexpr int max_q = (1 << 8);

typedef bvh::Bvh<float> bvh_t;
typedef bvh::Triangle<float> triangle_t;
typedef bvh::Vector3<float> vector_t;
typedef bvh::BoundingBox<float> bbox_t;
typedef bvh::Ray<float> ray_t;
typedef bvh::SweepSahBuilder<bvh_t> builder_t;
typedef bvh_t::Node node_t;
typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, triangle_t> primitive_intersector_t;

struct arg_t {
    char* model_file;
    float t_trv_int;
    float t_switch;
    float t_ist;
    char* ray_file;
};

enum policy_t {
    STAY, SWITCH
};

struct int_node_t {
    uint8_t bounds[6];
    // TODO: here!!
    unsigned int num_trigs : 3;
    unsigned int idx : 13;
};

struct int_cluster_t {
    float ref_bounds[6];
    float sx;
    int_node_t* local_nodes;
};

struct int_bvh_t {
    // clusters[0] is the top cluster
    std::unique_ptr<int_node_t[]> nodes;
    std::unique_ptr<int_cluster_t[]> clusters;
};

struct intersection_result_t {
    size_t triangle_idx;
    triangle_t::Intersection intersection;
};

struct statistics_t {
    traverser_t::Statistics s;
    size_t clusters = 0;
    size_t recompute_qymax = 0;
    size_t intersections_b = 0;
};

int floor_to_int(float x) {
    assert(!std::isnan(x));
    if (x < -2147483648.0f)
        return -2147483648;
    if (x >= 2147483648.0f)
        return 2147483647;
    assert(x >= -2147483648.0f && x < 2147483648.0f);
    return (int)floorf(x);
}

int ceil_to_int(float x) {
    assert(!std::isnan(x));
    if (x <= -2147483649.0f)
        return -2147483648;
    if (x > 2147483647.0f)
        return 2147483647;
    assert(x > -2147483649.0f && x <= 2147483647.0f);
    return (int)ceilf(x);
}

float get_scaling_factor(const bvh_t &bvh, size_t ref_idx) {
    bbox_t quant_bbox = bvh.nodes[ref_idx].bounding_box_proxy().to_bounding_box();

    float max_len = 0.0f;
    for (int i = 0; i < 3; i++)
        max_len = std::max(max_len, quant_bbox.max[i] - quant_bbox.min[i]);

    float scaling_factor = max_len / (float)quant_num;
    assert(scaling_factor > 0.0f);
    return scaling_factor;
}

// return: [Qxmin, Qxmax, Qymin, Qymax, Qzmin, Qzmax]
std::array<uint8_t, 6> get_quant_val(const bvh_t &bvh, size_t node_idx, size_t ref_idx, float scaling_factor) {
    bbox_t node_bbox = bvh.nodes[node_idx].bounding_box_proxy().to_bounding_box();
    bbox_t quant_bbox = bvh.nodes[ref_idx].bounding_box_proxy().to_bounding_box();

    std::array<uint8_t, 6> ret{};
    for (int i = 0; i < 3; i++) {
        int min = floor_to_int((node_bbox.min[i] - quant_bbox.min[i]) / scaling_factor);
        int max = ceil_to_int((node_bbox.max[i] - quant_bbox.min[i]) / scaling_factor);
        assert(0 <= min && min <= max_q);
        assert(0 <= max && max <= max_q);
        if (min == max_q)
            min = max_q - 1;
        if (max == max_q)
            max = max_q - 1;
        ret[i * 2] = min;
        ret[i * 2 + 1] = max;
    }

    return ret;
}

bbox_t get_quant_bbox(const bvh_t &bvh, size_t node_idx, size_t ref_idx) {
    bbox_t quant_bbox = bvh.nodes[ref_idx].bounding_box_proxy().to_bounding_box();

    float scaling_factor = get_scaling_factor(bvh, ref_idx);
    std::array<uint8_t, 6> quant_val = get_quant_val(bvh, node_idx, ref_idx, scaling_factor);

    bbox_t ret;
    for (int i = 0; i < 3; i++) {
        ret.min[i] = quant_bbox.min[i] + scaling_factor * (float)quant_val[i * 2];
        ret.max[i] = quant_bbox.min[i] + scaling_factor * (float)quant_val[i * 2 + 1];
        assert(std::isfinite(ret.min[i]));
        assert(std::isfinite(ret.max[i]));
    }
    return ret;
}

arg_t parse_arg(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] <<
                  " MODEL_FILE T_TRV_INT T_SWITCH T_IST [RAY_FILE]" << std::endl;
        exit(EXIT_FAILURE);
    }

    arg_t arg{};
    arg.model_file = argv[1];
    arg.t_trv_int = std::stof(argv[2]);
    arg.t_switch = std::stof(argv[3]);
    arg.t_ist = std::stof(argv[4]);
    std::cout << "MODEL_FILE = " << arg.model_file << std::endl;
    std::cout << "T_TRV_INT = " << arg.t_trv_int << std::endl;
    std::cout << "T_SWITCH = " << arg.t_switch << std::endl;
    std::cout << "T_IST = " << arg.t_ist << std::endl;

    arg.ray_file = nullptr;
    if (argc >= 6) {
        arg.ray_file = argv[5];
        std::cout << "RAY_FILE = " << arg.ray_file << std::endl;
    }
}

std::vector<triangle_t> load_triangles(const char* model_file) {
    happly::PLYData ply_data(model_file);
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<triangle_t> triangles;
    for (auto &face : f_idx) {
        triangles.emplace_back(
            vector_t((float)v_pos[face[0]][0], (float)v_pos[face[0]][1], (float)v_pos[face[0]][2]),
            vector_t((float)v_pos[face[1]][0], (float)v_pos[face[1]][1], (float)v_pos[face[1]][2]),
            vector_t((float)v_pos[face[2]][0], (float)v_pos[face[2]][1], (float)v_pos[face[2]][2])
        );
    }
    return triangles;
}

bvh_t build_bvh(std::vector<triangle_t> triangles) {
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    std::cout << "global_bbox = ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    bvh_t bvh;
    builder_t builder(bvh);
    builder.max_leaf_size = max_leaf_size;
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    return bvh;
}

std::vector<policy_t> get_policy(float t_trv_int, float t_switch, float t_ist, const bvh_t& bvh) {
    // stk_1: fill t_buf_size, t_buf_map
    int t_buf_size = 0;
    std::vector<int> t_buf_map(bvh.node_count);
    std::stack<std::pair<size_t, int>> stk_1;
    node_t &root_node = bvh.nodes[0];
    size_t root_left_node_idx = root_node.first_child_or_primitive;
    size_t root_right_node_idx = root_left_node_idx + 1;
    stk_1.emplace(root_right_node_idx, 1);
    stk_1.emplace(root_left_node_idx, 1);
    while (!stk_1.empty()) {
        auto [curr_idx, depth] = stk_1.top();
        node_t &curr_node = bvh.nodes[curr_idx];
        stk_1.pop();

        t_buf_map[curr_idx] = t_buf_size;
        t_buf_size += depth;

        if (!curr_node.is_leaf()) {
            size_t left_node_idx = curr_node.first_child_or_primitive;
            size_t right_node_idx = left_node_idx + 1;
            stk_1.emplace(right_node_idx, depth + 1);
            stk_1.emplace(left_node_idx, depth + 1);
        }
    }

    // stk_2: fill parent, t_buf, t_policy
    std::vector<size_t> parent(bvh.node_count);
    parent[root_left_node_idx] = 0;
    parent[root_right_node_idx] = 0;
    std::vector<float> t_buf(t_buf_size);
    std::vector<policy_t> t_policy(t_buf_size);
    std::stack<std::pair<size_t, bool>> stk_2;
    stk_2.emplace(root_right_node_idx, true);
    stk_2.emplace(root_left_node_idx, true);
    while (!stk_2.empty()) {
        auto [curr_idx, first] = stk_2.top();
        node_t &curr_node = bvh.nodes[curr_idx];
        stk_2.pop();

        size_t left_node_idx = curr_node.first_child_or_primitive;
        size_t right_node_idx = left_node_idx + 1;

        if (first) {
            stk_2.emplace(curr_idx, false);
            if (!curr_node.is_leaf()) {
                stk_2.emplace(right_node_idx, true);
                stk_2.emplace(left_node_idx, true);
                parent[left_node_idx] = curr_idx;
                parent[right_node_idx] = curr_idx;
            }
        } else {
            float left_switch_t;
            float right_switch_t;

            if (!curr_node.is_leaf()) {
                left_switch_t = t_buf[t_buf_map[left_node_idx]];
                right_switch_t = t_buf[t_buf_map[right_node_idx]];
            }

            size_t ref_idx = parent[curr_idx];
            for (int i = 0; ; i++) {
                bbox_t quant_bbox = get_quant_bbox(bvh, curr_idx, ref_idx);
                float half_area = quant_bbox.half_area();

                float &curr_t_buf = t_buf[t_buf_map[curr_idx] + i];
                policy_t &curr_t_policy = t_policy[t_buf_map[curr_idx] + i];
                if (curr_node.is_leaf()) {
                    curr_t_buf = t_ist * (float)curr_node.primitive_count * half_area;
                } else {
                    float left_stay_t = t_buf[t_buf_map[left_node_idx] + 1 + i];
                    float right_stay_t = t_buf[t_buf_map[right_node_idx] + 1 + i];

                    float curr_stay_t = t_trv_int * 2 * half_area + left_stay_t + right_stay_t;
                    float curr_switch_t = (t_trv_int * 2 + t_switch) * half_area + left_switch_t + right_switch_t;

                    assert(std::isfinite(curr_stay_t));
                    assert(std::isfinite(curr_switch_t));

                    if (curr_switch_t < curr_stay_t) {
                        curr_t_buf = curr_switch_t;
                        curr_t_policy = SWITCH;
                    } else {
                        curr_t_buf = curr_stay_t;
                        curr_t_policy = STAY;
                    }
                }

                if (ref_idx == 0)
                    break;
                else
                    ref_idx = parent[ref_idx];
            }
        }
    }

    // stk_3: fill policy
    std::vector<policy_t> policy(bvh.node_count);
    std::stack<std::pair<size_t, int>> stk_3;
    policy[0] = SWITCH;
    stk_3.emplace(root_right_node_idx, 0);
    stk_3.emplace(root_left_node_idx, 0);
    while (!stk_3.empty()) {
        auto [curr_idx, curr_offset] = stk_3.top();
        node_t& curr_node = bvh.nodes[curr_idx];
        stk_3.pop();

        if (curr_node.is_leaf())
            continue;

        size_t left_node_idx = curr_node.first_child_or_primitive;
        size_t right_node_idx = left_node_idx + 1;

        switch (t_policy[t_buf_map[curr_idx] + curr_offset]) {
            case STAY:
                policy[curr_idx] = STAY;
                stk_3.emplace(right_node_idx, 1 + curr_offset);
                stk_3.emplace(left_node_idx, 1 + curr_offset);
                break;
            case SWITCH:
                policy[curr_idx] = SWITCH;
                stk_3.emplace(right_node_idx, 0);
                stk_3.emplace(left_node_idx, 0);
                break;
        }
    }

    return policy;
}

int_bvh_t build_int_bvh(const bvh_t& bvh, const std::vector<policy_t>& policy) {
    // fill num_clusters, cluster_node_indices, ref_indices
    int num_clusters = 1;
    std::vector<std::vector<size_t>> cluster_node_indices(1);
    std::vector<size_t> ref_indices{0};
    std::queue<std::tuple<size_t, int, int>> que;
    node_t& root_node = bvh.nodes[0];
    size_t root_left_node_idx = root_node.first_child_or_primitive;
    size_t root_right_node_idx = root_left_node_idx + 1;
    que.emplace(root_left_node_idx, 0, 1);
    que.emplace(root_right_node_idx, 0, 1);
    while (!que.empty()) {
        auto [curr_idx, curr_cluster, curr_depth] = que.front();
        node_t &curr_node = bvh.nodes[curr_idx];
        que.pop();

        cluster_node_indices[curr_cluster].push_back(curr_idx);

        if (!curr_node.is_leaf()) {
            size_t left_idx = curr_node.first_child_or_primitive;
            size_t right_idx = left_idx + 1;

            int child_cluster;
            switch (policy[curr_idx]) {
                case STAY:
                    child_cluster = curr_cluster;
                    break;
                case SWITCH:
                    child_cluster = num_clusters++;
                    cluster_node_indices.emplace_back();
                    ref_indices.push_back(curr_idx);
                    break;
            }

            int child_depth = curr_depth + 1;
            que.emplace(left_idx, child_cluster, child_depth);
            que.emplace(right_idx, child_cluster, child_depth);
        }
    }

    auto cluster_map = std::make_unique<int[]>(bvh.node_count);
    auto scaling_factors = std::make_unique<float[]>(cluster_node_indices.size());
    std::fill(cluster_map.get(), cluster_map.get() + bvh.node_count, -1);
    for (int i = 0; i < cluster_node_indices.size(); i++) {
        for (int j = 0; j < cluster_node_indices[i].size(); j++)
            cluster_map[cluster_node_indices[i][j]] = i;
        scaling_factors[i] = get_scaling_factor(bvh, ref_indices[i]);
    }

    int_bvh_t int_bvh;
    int_bvh.nodes = std::make_unique<int_node_t[]>(bvh.node_count);
    int_bvh.clusters = std::make_unique<int_cluster_t[]>(num_clusters);
    int_node_t* tmp_nodes = int_bvh.nodes.get();
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < 6; j++)
            int_bvh.clusters[i].ref_bounds[j] = bvh.nodes[ref_indices[i]].bounds[j];
        int_bvh.clusters[i].sx = get_scaling_factor(bvh, ref_indices[i]);
        int_bvh.clusters[i].local_nodes = tmp_nodes;
        tmp_nodes += cluster_node_indices[i].size();

        // fill local_nodes
        for (int j = 0; j < cluster_node_indices[i].size(); j++) {
            std::array<uint8_t, 6> bounds = get_quant_val(bvh, cluster_node_indices[i][j], ref_indices[i],
                                                          scaling_factors[i]);
            for (int k = 0; k < 6; k++)
                int_bvh.clusters[i].local_nodes[j].bounds[k] = bounds[k];
        }
    }

    return int_bvh;
}

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