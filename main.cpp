#include <iostream>
#include <numeric>
#include <vector>
#include <unordered_map>
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

struct int_statistics_t {
    intmax_t traversal_steps = 0;
    intmax_t intersections = 0;
};

int get_quant_num(int quant_bits) {
    return (1 << quant_bits) - 2;
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

// return: [Zx, Zy, Zz]
std::array<int, 3> get_zero_point(const bvh_t &bvh, size_t quant_idx, float scaling_factor) {
    bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();

    std::array<int, 3> ret{};
    for (int i = 0; i < 3; i++)
        ret[i] = (int)floorf(quant_bbox.min[i] / scaling_factor);
    return ret;
}

// TODO: will double be better?
// return: [Qxmin, Qxmax, Qymin, Qymax, Qzmin, Qzmax]
std::array<int, 6> get_quant_val(const bvh_t &bvh, size_t node_idx, float scaling_factor, int quant_bits,
                                 const std::array<int, 3> &zero_point) {
    bbox_t node_bbox = bvh.nodes[node_idx].bounding_box_proxy().to_bounding_box();

    std::array<int, 6> ret{};
    for (int i = 0; i < 3; i++) {
        ret[i * 2] = (int)floorf(node_bbox.min[i] / scaling_factor) - zero_point[i];
        ret[i * 2 + 1] = (int)ceilf(node_bbox.max[i] / scaling_factor) - zero_point[i];
        int max_q = (1 << quant_bits);
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
    int quant_num = get_quant_num(quant_bits);
    float scaling_factor = get_scaling_factor(bvh, quant_idx, quant_num);
    std::array<int, 3> zero_point = get_zero_point(bvh, quant_idx, scaling_factor);
    std::array<int, 6> quant_val = get_quant_val(bvh, node_idx, scaling_factor, quant_bits, zero_point);

    bbox_t ret;
    for (int i = 0; i < 3; i++) {
        ret.min[i] = scaling_factor * (float)(quant_val[i * 2] + zero_point[i]);
        ret.max[i] = scaling_factor * (float)(quant_val[i * 2 + 1] + zero_point[i]);
        assert(std::isfinite(ret.min[i]));
        assert(std::isfinite(ret.max[i]));
    }
    return ret;
}

std::optional<intersection_result_t> intersect_leaf(ray_t& ray, size_t node_idx, int_statistics_t& statistics,
                                                    const bvh_t& bvh,
                                                    const std::vector<triangle_t>& triangles) {
    node_t& node = bvh.nodes[node_idx];
    assert(node.is_leaf());
    size_t begin = node.first_child_or_primitive;
    size_t end = begin + node.primitive_count;

    std::optional<intersection_result_t> best_hit;
    for (size_t i = begin; i < end; i++) {
        statistics.intersections++;
        size_t triangle_idx = bvh.primitive_indices[i];
        if (auto hit = triangles[triangle_idx].intersect(ray)) {
            best_hit = {triangle_idx, hit.value()};
            ray.tmax = hit->t;
        }
    }
    return best_hit;
}

std::pair<int, int> intersect_bbox(int qy_max,
                                   const std::array<int, 3>& qw,
                                   const std::array<int, 6>& qx,
                                   const std::array<int, 3>& zx,
                                   const std::array<int, 3>& qb) {
    const int& qx_x_a = qw[0] < 0 ? qx[1] : qx[0];
    const int& qx_x_b = qw[0] < 0 ? qx[0] : qx[1];
    const int& qx_y_a = qw[1] < 0 ? qx[3] : qx[2];
    const int& qx_y_b = qw[1] < 0 ? qx[2] : qx[3];
    const int& qx_z_a = qw[2] < 0 ? qx[5] : qx[4];
    const int& qx_z_b = qw[2] < 0 ? qx[4] : qx[5];

    int qxz_x_a = qx_x_a + zx[0];
    int qxz_x_b = qx_x_b + zx[0];
    int qxz_y_a = qx_y_a + zx[1];
    int qxz_y_b = qx_y_b + zx[1];
    int qxz_z_a = qx_z_a + zx[2];
    int qxz_z_b = qx_z_b + zx[2];

    int entry[3];
    int exit[3];
    entry[0] = std::min(qw[0] * qxz_x_a, (qw[0] + 1) * qxz_x_a) + qb[0];
    entry[1] = std::min(qw[1] * qxz_y_a, (qw[1] + 1) * qxz_y_a) + qb[1];
    entry[2] = std::min(qw[2] * qxz_z_a, (qw[2] + 1) * qxz_z_a) + qb[2];
    exit[0] = std::max(qw[0] * qxz_x_b, (qw[0] + 1) * qxz_x_b) + (qb[0] + 1);
    exit[1] = std::max(qw[1] * qxz_y_b, (qw[1] + 1) * qxz_y_b) + (qb[1] + 1);
    exit[2] = std::max(qw[2] * qxz_z_b, (qw[2] + 1) * qxz_z_b) + (qb[2] + 1);

    std::pair<int, int> ret;
    ret.first = std::max(entry[0], std::max(entry[1], std::max(entry[2], 0)));
    ret.second = std::min(exit[0], std::min(exit[1], std::min(exit[2], qy_max)));
    return ret;
}

int convert_to_int(float x) {
    assert(!std::isnan(x));
    if (x < -2147483648.0)
        return -2147483648;
    if (x >= 2147483648.0)
        return 2147483647;
    assert(x >= -2147483648.0 && x < 2147483648.0);
    return (int)floorf(x);
}

std::optional<intersection_result_t> int_traverse(ray_t& ray, float sw,
                                                  int_statistics_t& statistics,
                                                  const bvh_t& bvh,
                                                  const std::vector<triangle_t>& triangles,
                                                  const quant_node_t* quant_nodes,
                                                  const float* scaling_factors,
                                                  const std::array<int, 3>* zero_points,
                                                  const bvh_t& quant_bvh) {
    assert(ray.tmin == 0.0f);

    bvh::FastNodeIntersector<bvh_t> node_intersector(ray);
    intmax_t exact = 0;
    intmax_t redundant = 0;
    intmax_t error = 0;

    std::array<int, 3> qw{};
    for (int i = 0; i < 3; i++)
        qw[i] = convert_to_int(1.0f / (ray.direction[i] * sw));

    std::optional<intersection_result_t> best_hit;
    std::stack<size_t> stk;
    size_t left_idx = bvh.nodes[0].first_child_or_primitive;
    while (true) {
        statistics.traversal_steps++;

        size_t right_idx = left_idx + 1;
        int cluster_idx = quant_nodes[left_idx].cluster_idx;
        assert(cluster_idx == quant_nodes[right_idx].cluster_idx);

        const float& sx = scaling_factors[cluster_idx];
        const std::array<int, 3>& zx = zero_points[cluster_idx];
        std::array<int, 3> qb{};
        for (int i = 0; i < 3; i++)
            qb[i] = convert_to_int(-ray.origin[i] / (ray.direction[i] * sw * sx));
        int qy_max = convert_to_int(ray.tmax / (sw * sx));

        std::pair<int, int> distance_left = intersect_bbox(qy_max, qw, quant_nodes[left_idx].bounds, zx, qb);
        std::pair<int, int> distance_right = intersect_bbox(qy_max, qw, quant_nodes[right_idx].bounds, zx, qb);

        std::pair<float, float> ref_distance_left = node_intersector.intersect(quant_bvh.nodes[left_idx], ray);
        std::pair<float, float> ref_distance_right = node_intersector.intersect(quant_bvh.nodes[right_idx], ray);

        if (ref_distance_left.first <= ref_distance_left.second) {
            if (distance_left.first > distance_left.second)
                error++;
            else
                exact++;
        } else {
            if (distance_left.first <= distance_left.second)
                redundant++;
            else
                exact++;
        }

        if (ref_distance_right.first <= ref_distance_right.second) {
            if (distance_right.first > distance_right.second)
                error++;
            else
                exact++;
        } else {
            if (distance_right.first <= distance_right.second)
                redundant++;
            else
                exact++;
        }

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
                if (distance_left.first > distance_right.first)
                    std::swap(left_idx, right_idx);
                stk.emplace(bvh.nodes[right_idx].first_child_or_primitive);
            }
            left_idx = bvh.nodes[left_idx].first_child_or_primitive;
        } else if (right_hit) {
            left_idx = bvh.nodes[right_idx].first_child_or_primitive;
        } else {
            if (stk.empty())
                break;
            left_idx = stk.top();
            stk.pop();
        }
    }

    return best_hit;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "usage: ./a.out MODEL_FILE QUANT_BITS T_TRV T_SWITCH T_IST [RAY_FILE]" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *model_file = argv[1];
    int quant_bits = std::stoi(argv[2]);
    float t_trv = std::stof(argv[3]);
    float t_switch = std::stof(argv[4]);
    float t_ist = std::stof(argv[5]);
    std::cout << "MODEL_FILE = " << model_file << std::endl;
    std::cout << "QUANT_BITS = " << quant_bits << std::endl;
    std::cout << "T_TRV = " << t_trv << std::endl;
    std::cout << "T_SWITCH = " << t_switch << std::endl;
    std::cout << "T_IST = " << t_ist << std::endl;

    char *ray_file = nullptr;
    if (argc >= 7) {
        ray_file = argv[6];
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
        t_buf_size += depth;

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
    std::vector<char> t_stay(t_buf_size);
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
            size_t quant_idx = parent[curr_idx];
            for (int i = 0; ; i++) {
                bbox_t quant_bbox = get_quant_bbox(bvh, curr_idx, quant_idx, quant_bits);
                float half_area = quant_bbox.half_area();

                float &curr_t_buf = t_buf[t_buf_map[curr_idx] + i];
                char &curr_t_stay = t_stay[t_buf_map[curr_idx] + i];
                if (curr_node.is_leaf()) {
                    curr_t_buf = t_ist * (float)curr_node.primitive_count * half_area;
                } else {
                    float left_stay_t = t_buf[t_buf_map[left_node_idx] + 1 + i];
                    float right_stay_t = t_buf[t_buf_map[right_node_idx] + 1 + i];
                    float curr_stay_t = t_trv * 2 * half_area + left_stay_t + right_stay_t;

                    float left_switch_t = t_buf[t_buf_map[left_node_idx]];
                    float right_switch_t = t_buf[t_buf_map[right_node_idx]];
                    float curr_switch_t = (t_trv * 2 + t_switch) * half_area + left_switch_t + right_switch_t;

                    assert(std::isfinite(curr_stay_t));
                    assert(std::isfinite(curr_switch_t));
                    if (curr_stay_t < curr_switch_t) {
                        curr_t_buf = curr_stay_t;
                        curr_t_stay = true;
                    } else {
                        curr_t_buf = curr_switch_t;
                        curr_t_stay = false;
                    }
                }

                if (quant_idx == 0)
                    break;
                else
                    quant_idx = parent[quant_idx];
            }
        }
    }

    std::vector<bool> stay(bvh.node_count);
    stay[0] = true;
    std::stack<std::pair<size_t, int>> stk_3;
    stk_3.emplace(root_right_node_idx, 0);
    stk_3.emplace(root_left_node_idx, 0);
    while (!stk_3.empty()) {
        auto [curr_idx, curr_offset] = stk_3.top();
        node_t &curr_node = bvh.nodes[curr_idx];
        stk_3.pop();

        if (curr_node.is_leaf())
            continue;

        size_t left_node_idx = curr_node.first_child_or_primitive;
        size_t right_node_idx = left_node_idx + 1;

        if (t_stay[t_buf_map[curr_idx] + curr_offset]) {
            stay[curr_idx] = true;
            stk_3.emplace(right_node_idx, 1 + curr_offset);
            stk_3.emplace(left_node_idx, 1 + curr_offset);
        } else {
            stay[curr_idx] = false;
            stk_3.emplace(right_node_idx, 0);
            stk_3.emplace(left_node_idx, 0);
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

    std::array<std::string, 2> cmap = {"black", "red"};
    int num_clusters = 1;
    std::vector<std::vector<size_t>> cluster_indices(1);
    std::vector<size_t> quant_indices{0};
    std::queue<std::tuple<size_t, int, int, int>> que;
    que.emplace(0, 0, 0, 0);
    while (!que.empty()) {
        auto [curr_idx, curr_color, curr_cluster, curr_depth] = que.front();
        node_t &curr_node = bvh.nodes[curr_idx];
        que.pop();

        cluster_indices[curr_cluster].push_back(curr_idx);

        if (!curr_node.is_leaf()) {
            size_t left_idx = curr_node.first_child_or_primitive;
            size_t right_idx = left_idx + 1;

            int child_color;
            int child_cluster;
            if (stay[curr_idx]) {
                child_color = curr_color;
                child_cluster = curr_cluster;
            } else {
                child_color = (curr_color + 1) % 2;
                child_cluster = num_clusters++;
                cluster_indices.emplace_back();
                quant_indices.push_back(curr_idx);
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
    for (auto &x : cluster_indices) {
        bbox_t largest_bbox = bbox_t::empty();
        largest_bbox.extend(bvh.nodes[x[0]].bounding_box_proxy().to_bounding_box());
        largest_bbox.extend(bvh.nodes[x[1]].bounding_box_proxy().to_bounding_box());
        cluster_area.push_back(largest_bbox.volume());
    }
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

    if (ray_file == nullptr)
        return 0;

    std::unordered_map<size_t, int> cluster_map;
    float scaling_factors[cluster_indices.size()];
    std::array<int, 3> zero_points[cluster_indices.size()];
    for (int i = 0; i < cluster_indices.size(); i++) {
        int quant_num = get_quant_num(quant_bits);
        for (int j = 0; j < cluster_indices[i].size(); j++)
            cluster_map[cluster_indices[i][j]] = i;
        scaling_factors[i] = get_scaling_factor(bvh, quant_indices[i], quant_num);
        zero_points[i] = get_zero_point(bvh, quant_indices[i], scaling_factors[i]);
    }

    quant_node_t quant_nodes[bvh.node_count];
    for (int i = 0; i < bvh.node_count; i++) {
        quant_nodes[i].bounds = get_quant_val(bvh, i, scaling_factors[cluster_map[i]],
                                              quant_bits, zero_points[cluster_map[i]]);
        quant_nodes[i].cluster_idx = cluster_map[i];
    }

    std::cout << "traversing..." << std::endl;
    traverser_t traverser(bvh);
    traverser_t quant_traverser(quant_bvh);
    primitive_intersector_t primitive_intersector(bvh, triangles.data());
    primitive_intersector_t quant_primitive_intersector(quant_bvh, triangles.data());
    traverser_t::Statistics statistics;
    traverser_t::Statistics quant_statistics;
    int_statistics_t int_statistics;
    intmax_t total_rays = 0;
    intmax_t correct_rays = 0;
    std::ifstream ray_fs(ray_file);
    for (float r[7]; ray_fs.read((char*)r, 7 * sizeof(float)); total_rays++) {
        ray_t ray(
            bvh::Vector3<float>(r[0], r[1], r[2]),
            bvh::Vector3<float>(r[3], r[4], r[5]),
            0.f,
            r[6]
        );
        auto result = traverser.traverse(ray, primitive_intersector, statistics);
        auto quant_result = quant_traverser.traverse(ray, quant_primitive_intersector, quant_statistics);
        if (result.has_value())
            assert(quant_result.has_value() && quant_result->intersection.t <= result->intersection.t);
        else
            assert(!quant_result.has_value());

        auto int_result = int_traverse(ray, 0.01, int_statistics, bvh, triangles, quant_nodes,
                                       scaling_factors, zero_points, quant_bvh);
        if (result) {
            if (int_result &&
                result->primitive_index == int_result->triangle_idx &&
                result->intersection.t == int_result->intersection.t &&
                result->intersection.u == int_result->intersection.u &&
                result->intersection.v == int_result->intersection.v)
                correct_rays++;
        } else if (!int_result) {
            correct_rays++;
        }
    }
    std::cout << "traversal_steps: " << statistics.traversal_steps << std::endl;
    std::cout << "traversal_steps (quantized): " << quant_statistics.traversal_steps << std::endl;
    std::cout << "traversal_steps (int): " << int_statistics.traversal_steps << std::endl;
    std::cout << "intersections: " << statistics.intersections << std::endl;
    std::cout << "intersections (quantized): " << quant_statistics.intersections << std::endl;
    std::cout << "intersections (int): " << int_statistics.intersections << std::endl;
    std::cout << "total_rays: " << total_rays << std::endl;
    std::cout << "correct_rays: " << correct_rays << std::endl;
}
