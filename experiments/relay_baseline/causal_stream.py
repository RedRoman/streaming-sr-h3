"""Strictly causal protocol ablation for a pretrained RelayVSR decoder.

This is not the official algorithm: the current reference is repeated into both
endpoint slots, refreshed every K-1 frames, and decoder history resets there.
No future input frame is buffered or used. No weights are changed.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from refvsr_inference.runtime.streaming import CollaborativeFrameStream, OutputFrame, prepare_lr
from refvsr_inference.runtime.timing import DeviceSpan


class ReferenceAdapter(Protocol):
    """Optional project-local hook; no trained adapter is supplied by this file."""

    def reset(self) -> None: ...
    def observe(self, index: int, native_rgb: torch.Tensor, wan_latent: torch.Tensor) -> None: ...
    def read(self, index: int, current_latent: torch.Tensor) -> torch.Tensor: ...


class CausalRepeatStartStream(CollaborativeFrameStream):
    def __init__(self, *args, reference_adapter: ReferenceAdapter | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_adapter = reference_adapter

    @torch.no_grad()
    def frames(self, source):
        wan_state = self.pipeline.create_streaming_state(self.wan)
        generator = None if self.seed is None else [torch.Generator(device=self.device).manual_seed(self.seed)]
        self.last_keyframe_indices.clear()
        self.keyframe_count = 0
        if self.reference_adapter is not None:
            self.reference_adapter.reset()
        state = None
        shape = None
        expected_index = 0
        stride = 16 * max(int(self.wan.config.patch_size[1]), int(self.wan.config.patch_size[2])) // 4
        period = self.window_size - 1
        reference_index = 0
        for packet in source:
            if packet.index != expected_index:
                raise ValueError("Causal stream requires consecutive indices beginning at 0.")
            expected_index += 1
            if shape is None:
                shape = tuple(packet.rgb.shape)
                self.prepare(*shape[-2:])
            if tuple(packet.rgb.shape) != shape:
                raise ValueError("Causal stream requires constant frame dimensions.")
            timings = []
            span = DeviceSpan(self.device)
            native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
            timings.append(("flash_input", span.finish()))
            if packet.index % period == 0:
                reference_index = packet.index
                span = DeviceSpan(self.device)
                upsampled = F.interpolate(native, scale_factor=4, mode="bilinear", align_corners=False)
                timings.append(("wan_input", span.finish()))
                span = DeviceSpan(self.device)
                latent = self.pipeline.sample_streaming_latent_step(
                    model=self.wan, lq_frame=upsampled.unsqueeze(2), spatial_scale_factor=16,
                    prompt_embeds=self.prompt, streaming_state=wan_state, generator=generator,
                    device=self.device, compute_dtype=self.dtype,
                )[:, :, 0].to(self.dtype)
                timings.append(("wan", span.finish()))
                self.last_keyframe_indices.append(packet.index)
                self.keyframe_count += 1
                if self.reference_adapter is not None:
                    self.reference_adapter.observe(packet.index, packet.rgb, latent)
                    latent = self.reference_adapter.read(packet.index, latent)
                span = DeviceSpan(self.device)
                prepared = self.flash.prepare_condition(latent)
                state = self.flash.begin_stream(
                    prepared, prepared, total_frames=self.window_size,
                    input_height=native.shape[-2], input_width=native.shape[-1],
                )
                timings.append(("flash_condition", span.finish()))
            span = DeviceSpan(self.device)
            prediction, state = self.flash.decode_step(native, state)
            timings.append(("flash", span.finish()))
            h, w = shape[-2:]
            yield OutputFrame(packet.index, prediction[..., :h * 4, :w * 4],
                              packet.source_time, packet.arrival_time,
                              reference_index, reference_index, tuple(timings))
        if shape is None:
            raise ValueError("Cannot infer an empty causal stream.")
