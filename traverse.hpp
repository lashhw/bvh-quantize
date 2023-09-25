#ifndef TRAVERSE_HPP
#define TRAVERSE_HPP

#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>

constexpr auto inv_sw = static_cast<float>(1 << 7);

typedef bvh::Ray<float> ray_t;
typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, trig_t> primitive_intersector_t;
typedef trig_t::Intersection intersection_t;

struct cluster_data_t {
    uint16_t cluster_idx;
    int_node_t* local_nodes;
    trig_t* local_trigs;
    float inv_sx;
    float y_ref;
    int32_t qy_max;
    float o_local[3];
    int32_t qb_l[3];
    int32_t qb_h[3];
};

std::optional<intersection_t> intersect_leaf(const int_bvh_t& int_bvh, const decoded_data_t& decoded_data,
                                                    trig_t* local_trigs, ray_t& ray) {
    assert(decoded_data.child_type == child_type_t::LEAF);
    std::optional<intersection_t> best_hit;
    trig_t* tmp_trigs = &local_trigs[decoded_data.idx];
    for (size_t i = 0; i < decoded_data.num_trigs; i++) {
        if (auto hit = tmp_trigs->intersect(ray)) {
            best_hit = hit.value();
            ray.tmax = hit->t;
        }
        tmp_trigs++;
    }
    return best_hit;
}

struct int_w_t {
    bool iw[3];
    uint8_t rw_l[3];
    uint8_t qw_l[3];
    uint8_t rw_h[3];
    uint8_t qw_h[3];
};

int_w_t get_int_w(const std::array<float, 3>& w) {
    int_w_t int_w{};

    for (int i = 0; i < 3; i++) {
        float qw = floorf(inv_sw * w[i]);
        auto& qwi = reinterpret_cast<uint32_t&>(qw);

        bool sign = qwi & 0x80000000;
        uint32_t exponent = (qwi >> 23) & 0xff;
        uint32_t mantissa = qwi & 0x7fffff;

        // 127 + 7 = 134
        uint32_t low_exponent = (exponent - 134) & 0b11111;
        uint32_t low_mantissa = mantissa >> 16;
        uint32_t low = (low_exponent << 7) | low_mantissa;

        uint32_t high = sign ? (low - 1) : (low + 1);
        uint32_t high_exponent = high >> 7;
        uint32_t high_mantissa = high & 0b1111111;

        int_w.iw[i] = sign;
        int_w.rw_l[i] = low_exponent;
        int_w.qw_l[i] = 0b10000000 | low_mantissa;
        int_w.rw_h[i] = high_exponent;
        int_w.qw_h[i] = 0b10000000 | high_mantissa;
    }

    return int_w;
}

std::optional<float> intersect_full_bbox(const std::array<bool, 3>& octant,
                                         const std::array<float, 3>& w,
                                         const float* x,
                                         const std::array<float, 3>& b,
                                         float tmax) {
    float entry[3], exit[3];
    entry[0] = w[0] * x[0 + octant[0]] + b[0];
    entry[1] = w[1] * x[2 + octant[1]] + b[1];
    entry[2] = w[2] * x[4 + octant[2]] + b[2];
    exit [0] = w[0] * x[1 - octant[0]] + b[0];
    exit [1] = w[1] * x[3 - octant[1]] + b[1];
    exit [2] = w[2] * x[5 - octant[2]] + b[2];

    float entry_ = fmaxf(entry[0], fmaxf(entry[1], fmaxf(entry[2], 0.0f)));
    float exit_ = fminf(exit[0], fminf(exit[1], fminf(exit[2], tmax)));

    if (entry_ <= exit_)
        return entry_;
    else
        return std::nullopt;
}

std::optional<cluster_data_t> get_cluster_data(const int_bvh_t& int_bvh,
                                               const std::array<bool, 3>& octant,
                                               const std::array<float, 3>& w,
                                               const std::array<float, 3>& b,
                                               const ray_t& ray,
                                               uint16_t cluster_idx) {
    int_cluster_t curr_cluster = int_bvh.clusters[cluster_idx];
    auto y_ref = intersect_full_bbox(octant, w, curr_cluster.ref_bounds, b, ray.tmax);
    if (!y_ref.has_value())
        return std::nullopt;
    cluster_data_t ret{};
    ret.cluster_idx = cluster_idx;
    ret.local_nodes = &int_bvh.nodes[curr_cluster.node_offset];
    ret.local_trigs = &int_bvh.trigs[curr_cluster.trig_offset];
    ret.inv_sx = int_bvh.inv_sx[cluster_idx];
    ret.y_ref = y_ref.value();
    ret.qy_max = ceil_to_int((ray.tmax - ret.y_ref) * ret.inv_sx * inv_sw);
    for (int i = 0; i < 3; i++)
        ret.o_local[i] = ray.origin[i] + y_ref.value() * ray.direction[i] - curr_cluster.ref_bounds[2 * i];
    for (int i = 0; i < 3; i++) {
        ret.qb_l[i] = floor_to_int(-ret.o_local[i] * w[i] * ret.inv_sx * inv_sw);
        ret.qb_h[i] = ret.qb_l[i] + 1;
    }
    return ret;
}

std::optional<int> intersect_bbox(int32_t qy_max,
                                  const int_w_t& int_w,
                                  const uint8_t* qx,
                                  const int32_t* qb_l,
                                  const int32_t* qb_h) {
    const int qx_a[3] = {
        int_w.iw[0] ? qx[1] : qx[0],
        int_w.iw[1] ? qx[3] : qx[2],
        int_w.iw[2] ? qx[5] : qx[4]
    };

    const int qx_b[3] = {
        int_w.iw[0] ? qx[0] : qx[1],
        int_w.iw[1] ? qx[2] : qx[3],
        int_w.iw[2] ? qx[4] : qx[5]
    };

    int entry[3];
    int exit[3];
    entry[0] = (int_w.iw[0] ? -1 : 1) * ((int_w.qw_l[0] * qx_a[0]) << int_w.rw_l[0]) + qb_l[0];
    entry[1] = (int_w.iw[1] ? -1 : 1) * ((int_w.qw_l[1] * qx_a[1]) << int_w.rw_l[1]) + qb_l[1];
    entry[2] = (int_w.iw[2] ? -1 : 1) * ((int_w.qw_l[2] * qx_a[2]) << int_w.rw_l[2]) + qb_l[2];
    exit[0] = (int_w.iw[0] ? -1 : 1) * ((int_w.qw_h[0] * qx_b[0]) << int_w.rw_h[0]) + qb_h[0];
    exit[1] = (int_w.iw[1] ? -1 : 1) * ((int_w.qw_h[1] * qx_b[1]) << int_w.rw_h[1]) + qb_h[1];
    exit[2] = (int_w.iw[2] ? -1 : 1) * ((int_w.qw_h[2] * qx_b[2]) << int_w.rw_h[2]) + qb_h[2];

    int entry_ = std::max(entry[0], std::max(entry[1], std::max(entry[2], 0)));
    int exit_ = std::min(exit[0], std::min(exit[1], std::min(exit[2], qy_max)));

    if (entry_ <= exit_)
        return entry_;
    else
        return std::nullopt;
}

std::pair<uint16_t, uint16_t> get_node_cluster_pair(const decoded_data_t& decoded_data,
                                                    uint32_t cluster_idx) {
    switch (decoded_data.child_type) {
        case child_type_t::INTERNAL:
            return {decoded_data.idx, cluster_idx};
        case child_type_t::LEAF:
            assert(false);
        case child_type_t::SWITCH:
            return {0, decoded_data.idx};
    }
    return {};
}

std::optional<intersection_t> int_traverse(const int_bvh_t& int_bvh, ray_t& ray) {
    std::optional<intersection_t> best_hit;

    // preprocess ray
    assert(ray.tmin == 0.0f);
    std::array<bool, 3> octant = {
        std::signbit(ray.direction[0]),
        std::signbit(ray.direction[1]),
        std::signbit(ray.direction[2])
    };
    std::array<float, 3> w = {
        1.0f / ray.direction[0],
        1.0f / ray.direction[1],
        1.0f / ray.direction[2]
    };
    std::array<float, 3> b = {
        -ray.direction[0] * w[0],
        -ray.direction[1] * w[1],
        -ray.direction[2] * w[2]
    };
    int_w_t int_w = get_int_w(w);

    std::stack<cluster_data_t> stk_1;
    std::stack<std::pair<uint16_t, uint16_t>> stk_2;  // [local_node_idx, cluster_idx]
    if (auto x = get_cluster_data(int_bvh, octant, w, b, ray, 0))
        stk_1.push(x.value());
    else
        return std::nullopt;

    uint32_t left_local_node_idx = 0;
    assert(int_bvh.clusters[0].node_offset == left_local_node_idx);
    auto update_node_and_cluster = [&](const decoded_data_t& decoded_data) -> bool {
        switch (decoded_data.child_type) {
            case child_type_t::INTERNAL:
                left_local_node_idx = decoded_data.idx;
                return true;
            case child_type_t::LEAF:
                assert(false);
            case child_type_t::SWITCH:
                left_local_node_idx = 0;
                if (auto x = get_cluster_data(int_bvh, octant, w, b, ray, decoded_data.idx)) {
                    stk_1.push(x.value());
                    return true;
                } else {
                    return false;
                }
        }
        return false;
    };

    while (true) {
        uint32_t right_node_idx = left_local_node_idx + 1;
        int_node_t* left_node = &stk_1.top().local_nodes[left_local_node_idx];
        int_node_t* right_node = left_node + 1;
        auto distance_left = intersect_bbox(stk_1.top().qy_max, int_w, left_node->bounds,
                                            stk_1.top().qb_l, stk_1.top().qb_h);
        auto distance_right = intersect_bbox(stk_1.top().qy_max, int_w, right_node->bounds,
                                             stk_1.top().qb_l, stk_1.top().qb_h);

        bool left_hit = false;
        bool right_hit = false;

        decoded_data_t left_decoded_data = decode_data(left_node->data);
        decoded_data_t right_decoded_data = decode_data(right_node->data);

        if (distance_left) {
            if (left_decoded_data.child_type == child_type_t::LEAF) {
                if (auto hit = intersect_leaf(int_bvh, left_decoded_data, stk_1.top().local_trigs, ray))
                    best_hit = hit;
            } else {
                left_hit = true;
            }
        }

        if (distance_right) {
            if (right_decoded_data.child_type == child_type_t::LEAF) {
                if (auto hit = intersect_leaf(int_bvh, right_decoded_data, stk_1.top().local_trigs, ray))
                    best_hit = hit;
            } else {
                right_hit = true;
            }
        }

        if (left_hit) {
            if (right_hit) {
                if (distance_left.value() > distance_right.value())
                    std::swap(left_local_node_idx, right_node_idx);
                stk_2.push(get_node_cluster_pair(right_decoded_data, stk_1.top().cluster_idx));
            }
            if (update_node_and_cluster(left_decoded_data))
                continue;
        } else if (right_hit) {
            if (update_node_and_cluster(right_decoded_data))
                continue;
        }

        if (stk_2.empty())
            break;

        if (stk_1.top().cluster_idx != stk_2.top().second)
            stk_1.pop();

        while (true) {
            if (stk_1.top().cluster_idx == stk_2.top().second) {
                left_local_node_idx = stk_2.top().first;
                stk_2.pop();
                break;
            } else if (auto x = get_cluster_data(int_bvh, octant, w, b, ray, stk_2.top().second)) {
                stk_1.push(x.value());
                left_local_node_idx = stk_2.top().first;
                stk_2.pop();
                break;
            } else {
                stk_2.pop();
                if (stk_2.empty())
                    goto end;
            }
        }
    }

    end:
    return best_hit;
}

#endif //TRAVERSE_HPP
