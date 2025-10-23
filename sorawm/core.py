from pathlib import Path
from typing import Callable

import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm
import gc

from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector


class SoraWM:
    def __init__(self):
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner()

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        gc_interval: int = 30,
        max_reuse_misses: int = 60,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",
        }

        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(
                int(int(input_video_loader.original_bitrate) * 1.2)
            )
        else:
            output_options["crf"] = "18"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        # 流式/小批处理：不缓存全量帧，边读边检测、修复、写出
        logger.debug(
            f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
        )
        last_bbox = None
        misses_in_row = 0
        batch_count = 0

        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Process video")
        ):
            # 检测当前帧水印
            det = self.detector.detect(frame)
            if det["detected"]:
                bbox = det["bbox"]
                last_bbox = bbox
                misses_in_row = 0
            else:
                # 未检测到时复用上一帧 bbox，最多连续复用 max_reuse_misses 帧
                if last_bbox is not None and misses_in_row < max_reuse_misses:
                    bbox = last_bbox
                    misses_in_row += 1
                else:
                    bbox = None

            # 清理/直通
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                cleaned_frame = self.cleaner.clean(frame, mask)
            else:
                cleaned_frame = frame

            process_out.stdin.write(cleaned_frame.tobytes())
            # 释放本轮临时变量引用
            if bbox is not None:
                del mask
            del cleaned_frame
            del frame

            # 进度：10% - 95% 随处理推进
            if progress_callback and idx % 10 == 0:
                progress = 10 + int((idx / max(1, total_frames)) * 85)
                progress_callback(min(progress, 95))

            # 小批量/垃圾回收
            batch_count += 1
            if batch_count % max(1, gc_interval) == 0:
                gc.collect()

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)
        # 最终强制回收
        gc.collect()

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        # Clean up temporary file
        temp_output_path.unlink()
        logger.info(f"Saved no watermark video with audio at: {output_video_path}")


if __name__ == "__main__":
    from pathlib import Path

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
