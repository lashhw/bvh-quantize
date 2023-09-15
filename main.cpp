#include <iostream>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include "happly/happly.h"

typedef bvh::Bvh<float> bvh_t;
typedef bvh::Triangle<float> triangle_t;
typedef bvh::Vector3<float> vector_t;
typedef bvh::BoundingBox<float> bbox_t;
typedef bvh::Ray<float> ray_t;
typedef bvh::SweepSahBuilder<bvh_t> builder_t;
typedef bvh_t::Node node_t;
typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, triangle_t> primitive_intersector_t;

struct quant_node_t {
    std::array<int, 6> bounds;
    int cluster_idx;
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

int get_quant_num(int quant_bits) {
    return (1 << quant_bits) - 1;
}

enum policy_t {
    STAY, SWITCH, FULL
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

float get_scaling_factor(const bvh_t &bvh, size_t quant_idx, int quant_num) {
    bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();

    float max_len = 0.0f;
    for (int i = 0; i < 3; i++)
        max_len = std::max(max_len, quant_bbox.max[i] - quant_bbox.min[i]);

    float scaling_factor = max_len / (float)quant_num;
    assert(scaling_factor > 0.0f);
    return scaling_factor;
}

// return: [Qxmin, Qxmax, Qymin, Qymax, Qzmin, Qzmax]
std::array<int, 6> get_quant_val(const bvh_t &bvh, size_t node_idx, size_t quant_idx, float scaling_factor,
                                 int quant_bits) {
    bbox_t node_bbox = bvh.nodes[node_idx].bounding_box_proxy().to_bounding_box();
    bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();

    std::array<int, 6> ret{};
    for (int i = 0; i < 3; i++) {
        ret[i * 2] = floor_to_int((node_bbox.min[i] - quant_bbox.min[i]) / scaling_factor);
        ret[i * 2 + 1] = ceil_to_int((node_bbox.max[i] - quant_bbox.min[i]) / scaling_factor);
        int max_q = (1 << quant_bits);
        // TODO: use double to prevent error
        assert(0 <= ret[i * 2] && ret[i * 2] <= max_q);
        assert(0 <= ret[i * 2 + 1] && ret[i * 2 + 1] <= max_q);
        if (ret[i * 2] == max_q)
            ret[i * 2] = max_q - 1;
        if (ret[i * 2 + 1] == max_q)
            ret[i * 2 + 1] = max_q - 1;
    }
    return ret;
}

bbox_t get_quant_bbox(const bvh_t &bvh, size_t node_idx, size_t quant_idx, int quant_bits) {
    bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();

    int quant_num = get_quant_num(quant_bits);
    float scaling_factor = get_scaling_factor(bvh, quant_idx, quant_num);
    std::array<int, 6> quant_val = get_quant_val(bvh, node_idx, quant_idx, scaling_factor, quant_bits);

    bbox_t ret;
    for (int i = 0; i < 3; i++) {
        ret.min[i] = quant_bbox.min[i] + scaling_factor * (float)quant_val[i * 2];
        ret.max[i] = quant_bbox.min[i] + scaling_factor * (float)quant_val[i * 2 + 1];
        assert(std::isfinite(ret.min[i]));
        assert(std::isfinite(ret.max[i]));
    }
    return ret;
}

std::optional<intersection_result_t> intersect_leaf(ray_t& ray, size_t node_idx, statistics_t& statistics,
                                                    const bvh_t& bvh,
                                                    const std::vector<triangle_t>& triangles) {
    node_t& node = bvh.nodes[node_idx];
    assert(node.is_leaf());
    size_t begin = node.first_child_or_primitive;
    size_t end = begin + node.primitive_count;

    std::optional<intersection_result_t> best_hit;
    for (size_t i = begin; i < end; i++) {
        statistics.s.intersections_a++;
        size_t triangle_idx = bvh.primitive_indices[i];
        if (auto hit = triangles[triangle_idx].intersect(ray)) {
            best_hit = {triangle_idx, hit.value()};
            ray.tmax = hit->t;
        }
    }
    return best_hit;
}

std::pair<int, int> intersect_bbox(int qy_max,
                                   const std::array<bool, 3>& iw,
                                   const std::array<int, 3>& rw_l,
                                   const std::array<int, 3>& qw_l,
                                   const std::array<int, 3>& rw_h,
                                   const std::array<int, 3>& qw_h,
                                   const std::array<int, 6>& qx,
                                   const std::array<int, 3>& qb_l,
                                   const std::array<int, 3>& qb_h) {
    const int qx_a[3] = {
        iw[0] ? qx[1] : qx[0],
        iw[1] ? qx[3] : qx[2],
        iw[2] ? qx[5] : qx[4]
    };

    const int qx_b[3] = {
        iw[0] ? qx[0] : qx[1],
        iw[1] ? qx[2] : qx[3],
        iw[2] ? qx[4] : qx[5]
    };

    int entry[3];
    int exit[3];
    entry[0] = (iw[0] ? -1 : 1) * ((qw_l[0] * qx_a[0]) << rw_l[0]) + qb_l[0];
    entry[1] = (iw[1] ? -1 : 1) * ((qw_l[1] * qx_a[1]) << rw_l[1]) + qb_l[1];
    entry[2] = (iw[2] ? -1 : 1) * ((qw_l[2] * qx_a[2]) << rw_l[2]) + qb_l[2];
    exit[0] = (iw[0] ? -1 : 1) * ((qw_h[0] * qx_b[0]) << rw_h[0]) + qb_h[0];
    exit[1] = (iw[1] ? -1 : 1) * ((qw_h[1] * qx_b[1]) << rw_h[1]) + qb_h[1];
    exit[2] = (iw[2] ? -1 : 1) * ((qw_h[2] * qx_b[2]) << rw_h[2]) + qb_h[2];

    std::pair<int, int> ret;
    ret.first = std::max(entry[0], std::max(entry[1], std::max(entry[2], 0)));
    ret.second = std::min(exit[0], std::min(exit[1], std::min(exit[2], qy_max)));
    return ret;
}

// return: (sign, low_exponent, low_mantissa, high_exponent, high_mantissa)
std::tuple<bool, int, int, int, int> transform_w(float w) {
    w = floorf(w);
    int wi = *reinterpret_cast<int*>(&w);

    bool sign = wi & (1 << 31);
    int exponent = (wi >> 23) & 0b011111111;
    int mantissa = wi & ((1 << 23) - 1);

    int low_exponent = (exponent - (127 + 7)) & 0b11111;
    int low_mantissa = mantissa >> 16;
    int low = (low_exponent << 7) | low_mantissa;

    int high = sign ? (low - 1) : (low + 1);
    int high_exponent = high >> 7;
    int high_mantissa = high & 0b1111111;

    low_mantissa |= 0b10000000;
    high_mantissa |= 0b10000000;

    return {sign, low_exponent, low_mantissa, high_exponent, high_mantissa};
}

std::optional<intersection_result_t> int_traverse(ray_t& ray,
                                                  statistics_t& statistics,
                                                  const bvh_t& bvh,
                                                  const std::vector<triangle_t>& triangles,
                                                  const quant_node_t* quant_nodes,
                                                  const float* scaling_factors,
                                                  const std::vector<size_t>& quant_indices) {
    static std::unordered_set<int> cluster_set;
    cluster_set.clear();
    assert(ray.tmin == 0.0f);

    bvh::FastNodeIntersector<bvh_t> node_intersector(ray);

    int inv_sw = 1 << 7;
    auto inv_sw_f = static_cast<float>(inv_sw);
    std::array<bool, 3> iw{};
    std::array<int, 3> rw_l{};
    std::array<int, 3> qw_l{};
    std::array<int, 3> rw_h{};
    std::array<int, 3> qw_h{};
    for (int i = 0; i < 3; i++) {
        auto result = transform_w(inv_sw_f / ray.direction[i]);
        iw[i] = std::get<0>(result);
        rw_l[i] = std::get<1>(result);
        qw_l[i] = std::get<2>(result);
        rw_h[i] = std::get<3>(result);
        qw_h[i] = std::get<4>(result);
    }

    float prev_tmax = ray.tmax;
    std::optional<intersection_result_t> best_hit;
    std::stack<size_t> stk;
    stk.push(bvh.nodes[0].first_child_or_primitive);
    while (!stk.empty()) {
        statistics.s.traversal_steps++;

        size_t left_idx = stk.top();
        stk.pop();

        size_t right_idx = left_idx + 1;
        int cluster_idx = quant_nodes[left_idx].cluster_idx;
        bool new_cluster = cluster_set.count(cluster_idx) == 0;
        cluster_set.insert(cluster_idx);
        assert(cluster_idx != -1);
        assert(cluster_idx == quant_nodes[right_idx].cluster_idx);

        const float& sx = scaling_factors[cluster_idx];
        size_t quant_idx = quant_indices[cluster_idx];

        std::pair<float, float> y_quant_pair = node_intersector.intersect(bvh.nodes[quant_idx], ray);
        if (y_quant_pair.first > y_quant_pair.second)
            continue;
        float y_quant = y_quant_pair.first;

        bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();
        std::array<float, 3> o_local = {
            ray.origin[0] + y_quant * ray.direction[0] - quant_bbox.min[0],
            ray.origin[1] + y_quant * ray.direction[1] - quant_bbox.min[1],
            ray.origin[2] + y_quant * ray.direction[2] - quant_bbox.min[2]
        };
        std::array<int, 3> qb_l{};
        std::array<int, 3> qb_h{};
        for (int i = 0; i < 3; i++) {
            qb_l[i] = floor_to_int(-o_local[i] / (ray.direction[i] * sx) * inv_sw_f);
            qb_h[i] = qb_l[i] + 1;
        }
        if (!new_cluster && ray.tmax != prev_tmax)
            statistics.recompute_qymax++;
        prev_tmax = ray.tmax;
        int qy_max = ceil_to_int((ray.tmax - y_quant) / sx * inv_sw_f);

        std::pair<int, int> distance_left = intersect_bbox(qy_max, iw, rw_l, qw_l, rw_h, qw_h,
                                                           quant_nodes[left_idx].bounds, qb_l, qb_h);
        std::pair<int, int> distance_right = intersect_bbox(qy_max, iw, rw_l, qw_l, rw_h, qw_h,
                                                            quant_nodes[right_idx].bounds, qb_l, qb_h);

        bool left_hit = false;
        bool right_hit = false;

        if (distance_left.first <= distance_left.second) {
            if (bvh.nodes[left_idx].is_leaf()) {
                if (auto hit = intersect_leaf(ray, left_idx, statistics, bvh, triangles))
                    best_hit = hit;
            } else {
                left_hit = true;
            }
        }

        if (distance_right.first <= distance_right.second) {
            if (bvh.nodes[right_idx].is_leaf()) {
                if (auto hit = intersect_leaf(ray, right_idx, statistics, bvh, triangles))
                    best_hit = hit;
            } else {
                right_hit = true;
            }
        }

        if (left_hit) {
            if (right_hit) {
                statistics.s.both_intersected++;
                if (distance_left.first > distance_right.first)
                    std::swap(left_idx, right_idx);
                stk.emplace(bvh.nodes[right_idx].first_child_or_primitive);
            }
            stk.emplace(bvh.nodes[left_idx].first_child_or_primitive);
        } else if (right_hit) {
            stk.emplace(bvh.nodes[right_idx].first_child_or_primitive);
        }
    }

    statistics.clusters += cluster_set.size();

    if (best_hit.has_value())
        statistics.s.finalize++;

    return best_hit;
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "usage: " << argv[0] <<
            " MODEL_FILE QUANT_BITS T_TRV_FLOAT T_TRV_INT T_SWITCH T_IST [RAY_FILE]" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *model_file = argv[1];
    int quant_bits = std::stoi(argv[2]);
    float t_trv_float = std::stof(argv[3]);
    float t_trv_int = std::stof(argv[4]);
    float t_switch = std::stof(argv[5]);
    float t_ist = std::stof(argv[6]);
    std::cout << "MODEL_FILE = " << model_file << std::endl;
    std::cout << "QUANT_BITS = " << quant_bits << std::endl;
    std::cout << "T_TRV_FLOAT = " << t_trv_float << std::endl;
    std::cout << "T_TRV_INT = " << t_trv_int << std::endl;
    std::cout << "T_SWITCH = " << t_switch << std::endl;
    std::cout << "T_IST = " << t_ist << std::endl;

    char *ray_file = nullptr;
    if (argc >= 8) {
        ray_file = argv[7];
        std::cout << "RAY_FILE = " << ray_file << std::endl;
    }

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

    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    std::cout << "global_bbox = ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    std::cout << "building..." << std::endl;
    bvh_t bvh;
    builder_t builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::cout << "clustering..." << std::endl;
    int t_buf_size = 0;
    std::vector<int> t_buf_map(bvh.node_count, -1);
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
        t_buf_size += depth + 1;

        if (!curr_node.is_leaf()) {
            size_t left_node_idx = curr_node.first_child_or_primitive;
            size_t right_node_idx = left_node_idx + 1;
            stk_1.emplace(right_node_idx, depth + 1);
            stk_1.emplace(left_node_idx, depth + 1);
        }
    }

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
            float left_full_t;
            float right_full_t;

            if (!curr_node.is_leaf()) {
                left_switch_t = t_buf[t_buf_map[left_node_idx] + 1];
                right_switch_t = t_buf[t_buf_map[right_node_idx] + 1];
                left_full_t = t_buf[t_buf_map[left_node_idx]];
                right_full_t = t_buf[t_buf_map[right_node_idx]];
            }

            float &full_t_buf = t_buf[t_buf_map[curr_idx]];
            policy_t &full_t_policy = t_policy[t_buf_map[curr_idx]];

            // fill t*
            float full_half_area = bvh.nodes[curr_idx].bounding_box_proxy().half_area();
            if (curr_node.is_leaf()) {
                full_t_buf = t_ist * (float)curr_node.primitive_count * full_half_area;
            } else {
                float curr_full_t = t_trv_float * 2 * full_half_area + left_full_t + right_full_t;
                float curr_switch_t = (t_trv_int * 2 + t_switch) * full_half_area + left_switch_t + right_switch_t;
                if (curr_full_t < curr_switch_t) {
                    full_t_buf = curr_full_t;
                    full_t_policy = FULL;
                } else {
                    full_t_buf = curr_switch_t;
                    full_t_policy = SWITCH;
                }
            }

            size_t quant_idx = parent[curr_idx];
            for (int i = 1; ; i++) {
                bbox_t quant_bbox = get_quant_bbox(bvh, curr_idx, quant_idx, quant_bits);
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
                    float curr_full_t = t_trv_float * 2 * half_area + left_full_t + right_full_t;

                    assert(std::isfinite(curr_stay_t));
                    assert(std::isfinite(curr_switch_t));
                    if (t_trv_float != std::numeric_limits<float>::infinity())
                        assert(std::isfinite(curr_full_t));

                    if (curr_full_t < curr_switch_t) {
                        if (curr_full_t < curr_stay_t) {
                            curr_t_buf = curr_full_t;
                            curr_t_policy = FULL;
                        } else {
                            curr_t_buf = curr_stay_t;
                            curr_t_policy = STAY;
                        }
                    } else {
                        if (curr_switch_t < curr_stay_t) {
                            curr_t_buf = curr_switch_t;
                            curr_t_policy = SWITCH;
                        } else {
                            curr_t_buf = curr_stay_t;
                            curr_t_policy = STAY;
                        }
                    }
                }

                if (quant_idx == 0)
                    break;
                else
                    quant_idx = parent[quant_idx];
            }
        }
    }

    float root_left_switch_t = t_buf[t_buf_map[root_left_node_idx] + 1];
    float root_right_switch_t = t_buf[t_buf_map[root_right_node_idx] + 1];
    float root_left_full_t = t_buf[t_buf_map[root_left_node_idx]];
    float root_right_full_t = t_buf[t_buf_map[root_right_node_idx]];
    float root_half_area = bvh.nodes[0].bounding_box_proxy().to_bounding_box().half_area();
    float root_switch_t = (t_trv_int * 2 + t_switch) * root_half_area + root_left_switch_t + root_right_switch_t;
    float root_full_t = t_trv_float * 2 * root_half_area + root_left_full_t + root_right_full_t;

    std::vector<policy_t> policy(bvh.node_count);
    std::stack<std::pair<size_t, int>> stk_3;
    if (root_full_t < root_switch_t) {
        policy[0] = FULL;
        stk_3.emplace(root_right_node_idx, 0);
        stk_3.emplace(root_left_node_idx, 0);
    } else {
        policy[0] = SWITCH;
        stk_3.emplace(root_right_node_idx, 1);
        stk_3.emplace(root_left_node_idx, 1);
    }
    while (!stk_3.empty()) {
        auto [curr_idx, curr_offset] = stk_3.top();
        node_t &curr_node = bvh.nodes[curr_idx];
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
                stk_3.emplace(right_node_idx, 1);
                stk_3.emplace(left_node_idx, 1);
                break;
            case FULL:
                policy[curr_idx] = FULL;
                stk_3.emplace(right_node_idx, 0);
                stk_3.emplace(left_node_idx, 0);
                break;
        }
    }

    std::cout << "output result..." << std::endl;
    std::ofstream graph_fs("graph.dot");
    graph_fs << "digraph bvh {\n";
    graph_fs << "    layout=twopi\n";
    graph_fs << "    ranksep=2\n";
    graph_fs << "    root=0\n";
    graph_fs << "    node [shape=point]\n";
    graph_fs << "    edge [arrowhead=none penwidth=0.5]\n";
    graph_fs << "    0 [shape=circle label=root]\n";

    std::array<std::string, 3> cmap = {"black", "red", "green"};
    int num_clusters = 0;
    std::vector<std::vector<size_t>> cluster_indices;
    std::vector<size_t> quant_indices;
    std::vector<size_t> full_indices;
    std::queue<std::tuple<size_t, int, int, int>> que;
    que.emplace(0, 0, -1, 0);
    while (!que.empty()) {
        auto [curr_idx, curr_color, curr_cluster, curr_depth] = que.front();
        node_t &curr_node = bvh.nodes[curr_idx];
        que.pop();

        if (curr_cluster == -1)
            full_indices.push_back(curr_idx);
        else
            cluster_indices[curr_cluster].push_back(curr_idx);

        if (!curr_node.is_leaf()) {
            size_t left_idx = curr_node.first_child_or_primitive;
            size_t right_idx = left_idx + 1;

            int child_color;
            int child_cluster;
            switch (policy[curr_idx]) {
                case STAY:
                    child_color = curr_color;
                    child_cluster = curr_cluster;
                    break;
                case SWITCH:
                    child_color = (curr_color + 1) % 2;
                    child_cluster = num_clusters++;
                    cluster_indices.emplace_back();
                    quant_indices.push_back(curr_idx);
                    break;
                case FULL:
                    child_color = 2;
                    child_cluster = -1;
                    break;
            }

            int child_depth = curr_depth + 1;
            graph_fs << "    " << left_idx << " [depth=" << child_depth << "]\n";
            graph_fs << "    " << right_idx << " [depth=" << child_depth << "]\n";
            graph_fs << "    " << curr_idx << " -> " << left_idx << " [color=" << cmap[child_color] << "]\n";
            graph_fs << "    " << curr_idx << " -> " << right_idx << " [color=" << cmap[child_color] << "]\n";

            que.emplace(left_idx, child_color, child_cluster, child_depth);
            que.emplace(right_idx, child_color, child_cluster, child_depth);
        }
    }

    graph_fs << "}";

    std::vector<float> cluster_area;
    for (const auto &x : quant_indices)
        cluster_area.push_back(bvh.nodes[x].bounding_box_proxy().to_bounding_box().half_area());
    std::vector<int> sorted_cluster_indices(cluster_area.size());
    std::iota(sorted_cluster_indices.begin(), sorted_cluster_indices.end(), 0);
    std::sort(sorted_cluster_indices.begin(), sorted_cluster_indices.end(), [&](int l, int r) {
        return cluster_area[l] > cluster_area[r];
    });

    bvh_t quant_bvh;
    quant_bvh.nodes = std::make_unique<node_t[]>(bvh.node_count);
    quant_bvh.primitive_indices = std::make_unique<size_t[]>(triangles.size());
    quant_bvh.node_count = bvh.node_count;
    std::copy(bvh.primitive_indices.get(), bvh.primitive_indices.get() + triangles.size(),
              quant_bvh.primitive_indices.get());

    std::ofstream cluster_size_fs("cluster_size.bin", std::ios::binary);
    std::ofstream bbox_fs("bbox.bin", std::ios::binary);
    std::ofstream quant_bbox_fs("quant_bbox.bin", std::ios::binary);
    for (const auto &cluster_idx : sorted_cluster_indices) {
        uint32_t size = cluster_indices[cluster_idx].size();
        cluster_size_fs.write((char*)(&size), sizeof(size));
        for (const auto &idx : cluster_indices[cluster_idx]) {
            for (float plane : bvh.nodes[idx].bounds)
                bbox_fs.write((char*)(&plane), sizeof(plane));
            bbox_t quant_bbox = get_quant_bbox(bvh, idx, quant_indices[cluster_idx], quant_bits);
            quant_bvh.nodes[idx].bounding_box_proxy() = quant_bbox;
            quant_bvh.nodes[idx].primitive_count = bvh.nodes[idx].primitive_count;
            quant_bvh.nodes[idx].first_child_or_primitive = bvh.nodes[idx].first_child_or_primitive;
            quant_bbox_fs.write((char*)(&quant_bbox.min[0]), sizeof(quant_bbox.min[0]));
            quant_bbox_fs.write((char*)(&quant_bbox.max[0]), sizeof(quant_bbox.max[0]));
            quant_bbox_fs.write((char*)(&quant_bbox.min[1]), sizeof(quant_bbox.min[1]));
            quant_bbox_fs.write((char*)(&quant_bbox.max[1]), sizeof(quant_bbox.max[1]));
            quant_bbox_fs.write((char*)(&quant_bbox.min[2]), sizeof(quant_bbox.min[2]));
            quant_bbox_fs.write((char*)(&quant_bbox.max[2]), sizeof(quant_bbox.max[2]));
        }
    }

    std::ofstream full_bbox_fs("full_bbox.bin", std::ios::binary);
    for (const auto &idx : full_indices) {
        quant_bvh.nodes[idx].bounding_box_proxy() = bvh.nodes[idx].bounding_box_proxy().to_bounding_box();
        quant_bvh.nodes[idx].primitive_count = bvh.nodes[idx].primitive_count;
        quant_bvh.nodes[idx].first_child_or_primitive = bvh.nodes[idx].first_child_or_primitive;
        for (float plane : bvh.nodes[idx].bounds)
            full_bbox_fs.write((char*)(&plane), sizeof(plane));
    }

    size_t num_quant_bbox = 0;
    for (const auto &x : cluster_indices)
        num_quant_bbox += x.size();
    std::cout << "# of clusters: " << cluster_indices.size() << std::endl;
    std::cout << "# of bboxes: " << bvh.node_count << std::endl;
    std::cout << "# of full bboxes: " << full_indices.size() << std::endl;
    std::cout << "# of quantized bboxes: " << num_quant_bbox << std::endl;
    assert(bvh.node_count == full_indices.size() + num_quant_bbox);

    if (ray_file == nullptr)
        return 0;

    auto cluster_map = std::make_unique<int[]>(bvh.node_count);
    float scaling_factors[cluster_indices.size()];
    std::fill(cluster_map.get(), cluster_map.get() + bvh.node_count, -1);
    for (int i = 0; i < cluster_indices.size(); i++) {
        int quant_num = get_quant_num(quant_bits);
        for (int j = 0; j < cluster_indices[i].size(); j++)
            cluster_map[cluster_indices[i][j]] = i;
        scaling_factors[i] = get_scaling_factor(bvh, quant_indices[i], quant_num);
    }

    auto quant_nodes = std::make_unique<quant_node_t[]>(bvh.node_count);
    for (int i = 0; i < bvh.node_count; i++) {
        if (cluster_map[i] == -1) {
            quant_nodes[i].cluster_idx = -1;
        } else {
            quant_nodes[i].bounds = get_quant_val(bvh, i, quant_indices[cluster_map[i]],
                                                  scaling_factors[cluster_map[i]], quant_bits);
            quant_nodes[i].cluster_idx = cluster_map[i];
        }
    }

    std::cout << "traversing..." << std::endl;
    traverser_t traverser(bvh);
    traverser_t quant_traverser(quant_bvh);
    primitive_intersector_t primitive_intersector(bvh, triangles.data());
    primitive_intersector_t quant_primitive_intersector(quant_bvh, triangles.data());
    statistics_t statistics;
    statistics_t quant_statistics;
    statistics_t int_statistics;
    intmax_t total_rays = 0;
    intmax_t correct_rays = 0;
    std::ifstream ray_fs(ray_file);
    for (float r[7]; ray_fs.read((char*)r, 7 * sizeof(float)); total_rays++) {
        ray_t ray(
            vector_t(r[0], r[1], r[2]),
            vector_t(r[3], r[4], r[5]),
            0.f,
            r[6]
        );
        ray_t ray_ = ray;
        bvh::intersections_b = &statistics.intersections_b;
        auto result = traverser.traverse(ray_, primitive_intersector, statistics.s);
        ray_ = ray;
        bvh::intersections_b = &quant_statistics.intersections_b;
        auto quant_result = quant_traverser.traverse(ray_, quant_primitive_intersector, quant_statistics.s);
        if (result.has_value())
            assert(quant_result.has_value() && quant_result->intersection.t <= result->intersection.t);
        else
            assert(!quant_result.has_value());

        ray_ = ray;
        bvh::intersections_b = &int_statistics.intersections_b;
        auto int_result = int_traverse(ray_, int_statistics, bvh, triangles, quant_nodes.get(),
                                       scaling_factors, quant_indices);
        if (result.has_value()) {
            if (int_result.has_value() &&
                result->primitive_index == int_result->triangle_idx &&
                result->intersection.t == int_result->intersection.t &&
                result->intersection.u == int_result->intersection.u &&
                result->intersection.v == int_result->intersection.v)
                correct_rays++;
        } else if (!int_result.has_value()) {
            correct_rays++;
        }
    }

    std::cout << "traversal_steps:" << std::endl;
    std::cout << "  (vanilla) " << statistics.s.traversal_steps << std::endl;
    std::cout << "  (quant) " << quant_statistics.s.traversal_steps << std::endl;
    std::cout << "  (int) " << int_statistics.s.traversal_steps << std::endl;

    std::cout << "intersections_a:" << std::endl;
    std::cout << "  (vanilla) " << statistics.s.intersections_a << std::endl;
    std::cout << "  (quant) " << quant_statistics.s.intersections_a << std::endl;
    std::cout << "  (int) " << int_statistics.s.intersections_a << std::endl;

    std::cout << "intersections_b:" << std::endl;
    std::cout << "  (vanilla) " << statistics.intersections_b << std::endl;
    std::cout << "  (quant) " << quant_statistics.intersections_b << std::endl;
    std::cout << "  (int) " << int_statistics.intersections_b << std::endl;

    std::cout << "both_intersected:" << std::endl;
    std::cout << "  (vanilla) " << statistics.s.both_intersected << std::endl;
    std::cout << "  (quant) " << quant_statistics.s.both_intersected << std::endl;
    std::cout << "  (int) " << int_statistics.s.both_intersected << std::endl;

    std::cout << "finalize:" << std::endl;
    std::cout << "  (vanilla) " << statistics.s.finalize << std::endl;
    std::cout << "  (quant) " << quant_statistics.s.finalize << std::endl;
    std::cout << "  (int) " << int_statistics.s.finalize << std::endl;

    std::cout << "clusters:" << std::endl;
    std::cout << "  (int) " << int_statistics.clusters << std::endl;

    std::cout << "recompute_qymax:" << std::endl;
    std::cout << "  (int) " << int_statistics.recompute_qymax << std::endl;

    std::cout << "total_rays: " << total_rays << std::endl;
    std::cout << "correct_rays: " << correct_rays << std::endl;
}
