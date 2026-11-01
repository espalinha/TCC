from fase1_entrada.segmentacao.stages import (
    absorb_defect_masks, clean_masks, complete_teeth, detect, drop_border_blobs, drop_off_arch,
    drop_opposite_arch, drop_outliers, drop_red_tissue, drop_tissue_shapes, filter_by_base_confidence,
    filter_by_confidence, resolve_shared, separate_fused, split_arches, whole_arch,
)


def lateral(detector):
    return [detect(detector), filter_by_base_confidence, clean_masks, drop_red_tissue, drop_outliers,
            absorb_defect_masks, complete_teeth, split_arches, resolve_shared, separate_fused]


def occlusal(detector):
    # forma de tecido ANTES de completar com o casco convexo (o casco deixa qualquer bloco "sólido")
    return [detect(detector), filter_by_confidence, clean_masks, drop_red_tissue, drop_outliers,
            drop_tissue_shapes, drop_border_blobs, drop_opposite_arch, drop_off_arch,
            absorb_defect_masks, complete_teeth, whole_arch]


def run(state, stages):
    for stage in stages:
        if state.stopped:
            break
        state = stage(state)
    return state
