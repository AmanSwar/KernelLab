import triton
import triton.language as tl

import torch
import math

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32


@triton.jit
def _self_attn_fwd(
    Q,  # query tensor [B , H , T , head_dim]
    Kt,  # key tensor transposed [B , H , head_dim , T]
    V,  # value tensor [B , H , T , head_dim]
    L,  # softmax denom
    O,  # output tensor
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qk,  # strides b/w batch , head , sequence lenght and head dim
    stride_kb,
    stride_kh,
    stride_kk,
    stride_kt,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_ok,
    lens_stride,  # stride for seq length
    T: int,  # total length of seq
    PRESCALE: tl.constexpr,  # queries are prescaled or not
    LEN_PRESENT: tl.constexpr,  # uf we have length masking or not
    HEAD_DIM: tl.constexpr,  # dim of head
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    PIPELINING: tl.constexpr,
    RCP_LN2: tl.constexpr, #reciprocal of ln(2) -> for fast log approx in softmax
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    if LEN_PRESENT:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len , T)
        need_q_mask = q_token_idx + TILE_Q_SIZE >= seq_len

    else:
        seq_len = T
        need_q_mask = False

    if seq_len <= q_tile_idx:
        return 

    qbatch_head_offset = batch * stride_qb + head * stride_qh

    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1 , 0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    q_tile = tl.load(
        q_tile_ptr,
        boundary_check=(0,),
    )
    softmax_scale = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE:
        q_tile *= softmax_scale

    max_tile = tl.cdiv(seq_len, TILE_K_SIZE)

    for kv_tile_idx in tl.range(0 , max_tile , num_stages=PIPELINING):
        last_iter = kv_tile_idx == max_tile - 1
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if last_iter:
            kt_tile = tl.load(
                    tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                    boundary_check=(1,),
                )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )

        if V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    boundary_check=(0,),
                )
            else:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                )

        qk = tl.dot(q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        if not PRESCALE:
            qk *= softmax_scale

        if last_iter:
            kv_indices = kv_token_idx + tile_k_arange

            mask = (
                kv_indices[None, :] < seq_len
            )

            qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if not V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                        tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                        boundary_check=(0,),
                    )
            else:
                v_tile = tl.load(
                        tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    )
        acc = tl.dot(
            p.to(v_tile.dtype),
            v_tile,
            acc,
            input_precision=INPUT_PRECISION,
            out_dtype=tl.float32,
        )
        m_i = m_ij

    acc = acc / l_i[:, None]
    if need_q_mask:
        q_lens_mask = q_tile_indices[:, None] < seq_len
        acc = tl.where(q_lens_mask, acc, 0.0)

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(
        o_tile_ptr,
        acc.to(o_tile_ptr.type.element_ty),
        boundary_check=(0,),
    )
