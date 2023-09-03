## Vanilla
### per traversal_steps
* 12 FP32_MUL
* 12 FP32_ADDSUB
* 14 FP32_CMP

### per both_intersected
* 1 FP32_CMP

## Quantized
### per clusters
* 11 FP32_MUL
* 12 FP32_ADDSUB
* 7 FP32_CMP

### per recompute_ymax
* 1 FP32_MUL
* 1 FP32_ADDSUB

### per traversal_steps
* 12 INT8_MUL
* 12 INT32_ADDSUB
* 14 INT32_CMP

### per both_intersected
* 1 INT32_CMP

## Common
### per ray
* 3 FP32_RCP
* 3 FP32_MUL

### per intersections_a
* 15 FP32_MUL
* 13 FP32_ADDSUB
* 1 FP32_CMP

### per intersections_b
* 5 FP32_MUL 
* 2 FP32_ADDSUB
* 2 FP32_CMP

### per finalize
* 1 FP32_RCP
* 3 FP32_MUL

```cpp
__forceinline bool intersect(const vbool<M>& valid0,
                             Ray& ray,
                             const Vec3vf<M>& tri_v0,
                             const Vec3vf<M>& tri_e1,
                             const Vec3vf<M>& tri_e2,
                             const Vec3vf<M>& tri_Ng,
                             const UVMapper& mapUV,
                             MoellerTrumboreHitM<M,UVMapper>& hit) const {
    /* calculate denominator */
    vbool<M> valid = valid0;
    
    const Vec3vf<M> O = Vec3vf<M>((Vec3fa)ray.org);
    const Vec3vf<M> D = Vec3vf<M>((Vec3fa)ray.dir);
    const Vec3vf<M> C = Vec3vf<M>(tri_v0) - O; // 3 SUB
    const Vec3vf<M> R = cross(C,D); // 6 MUL 3 SUB
    const vfloat<M> den = dot(Vec3vf<M>(tri_Ng),D); // 3 MUL 2 ADD
    const vfloat<M> absDen = abs(den);
    const vfloat<M> sgnDen = signmsk(den);

    /* perform edge tests */
    const vfloat<M> U = asFloat(asInt(dot(R,Vec3vf<M>(tri_e2))) ^ asInt(sgnDen));  // 3 MUL 2 ADD
    const vfloat<M> V = asFloat(asInt(dot(R,Vec3vf<M>(tri_e1))) ^ asInt(sgnDen));  // 3 MUL 2 ADD
    
    valid &= (den != vfloat<M>(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDen);  // 1 ADD 1 CMP
    if (likely(none(valid))) return false;

    /* perform depth test */
    const vfloat<M> T = asFloat(asInt(dot(Vec3vf<M>(tri_Ng),C)) ^ asInt(sgnDen));  // 3 MUL 2 ADD

    valid &= (absDen*vfloat<M>(ray.tnear()) < T) & (T <= absDen*vfloat<M>(ray.tfar));  // 2 MUL 2 CMP
    if (likely(none(valid))) return false;
        
    /* update hit information */
    new (&hit) MoellerTrumboreHitM<M,UVMapper>(valid,U,V,T,absDen,tri_Ng,mapUV);

    return true;
}
    
__forceinline void finalize() {
    const vfloat<M> rcpAbsDen = rcp(absDen);  // 1 RCP
    vt = T * rcpAbsDen;  // 1 MUL
    vu = U * rcpAbsDen;  // 1 MUL
    vv = V * rcpAbsDen;  // 1 MUL
    mapUV(vu,vv,vNg);
}
```
