#include <iostream>
#include <numeric>
#include <vector>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include "happly/happly.h"

typedef bvh::Bvh<float> bvh_t;
typedef bvh::Triangle<float> triangle_t;
typedef bvh::Vector3<float> vector_t;
typedef bvh::BoundingBox<float> bbox_t;
typedef bvh::SweepSahBuilder<bvh_t> builder_t;
typedef bvh_t::Node node_t;

bbox_t get_quant_bbox(bvh_t &bvh, size_t node_idx, size_t quant_idx, int quant_bits) {
    bbox_t node_bbox = bvh.nodes[node_idx].bounding_box_proxy().to_bounding_box();
    bbox_t quant_bbox = bvh.nodes[quant_idx].bounding_box_proxy().to_bounding_box();
    int quant_num = 1 << quant_bits;

    bbox_t ret;
    for (int i = 0; i < 3; i++) {
        if (quant_bbox.min[i] == quant_bbox.max[i]) {
            ret.min[i] = quant_bbox.min[i];
            ret.max[i] = quant_bbox.max[i];
        } else {
            float scaling_factor = (quant_bbox.max[i] - quant_bbox.min[i]) / (float)quant_num;
            ret.min[i] = quant_bbox.min[i] + scaling_factor * floorf((node_bbox.min[i] - quant_bbox.min[i]) / scaling_factor);
            ret.max[i] = quant_bbox.min[i] + scaling_factor * ceilf((node_bbox.max[i] - quant_bbox.min[i]) / scaling_factor);
        }
        assert(std::isfinite(ret.min[i]));
        assert(std::isfinite(ret.max[i]));
    }
    return ret;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "usage: ./a.out MODEL_FILE QUANT_BITS T_TRV T_SWITCH T_IST" << std::endl;
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
    std::stack<std::pair<size_t, int>> stk_1;
    std::vector<int> t_buf_map(bvh.node_count, -1);
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
    std::vector<std::vector<size_t>> cluster_indices(1);
    int num_clusters = 1;
    std::queue<std::tuple<size_t, int, int, int>> que;
    que.emplace(0, 0, 0, 0);
    for (int i = 0; !que.empty(); i++) {
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
            }

            int child_depth = curr_depth + 1;
            graph_fs << "    " << curr_idx << " -> " << left_idx << " [color=" << cmap[child_color] << "]\n";
            graph_fs << "    " << curr_idx << " -> " << right_idx << " [color=" << cmap[child_color] << "]\n";
            graph_fs << "    " << left_idx << " [depth=" << child_depth << "]\n";
            graph_fs << "    " << right_idx << " [depth=" << child_depth << "]\n";

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

    std::ofstream cluster_size_fs("cluster_size.bin", std::ios::binary);
    std::ofstream bbox_fs("bbox.bin", std::ios::binary);
    for (const auto &cluster_idx : sorted_cluster_indices) {
        uint32_t size = cluster_indices[cluster_idx].size();
        cluster_size_fs.write((char*)(&size), sizeof(size));
        for (const auto &idx : cluster_indices[cluster_idx])
            for (float plane : bvh.nodes[idx].bounds)
                bbox_fs.write((char*)(&plane), sizeof(plane));
    }
}
