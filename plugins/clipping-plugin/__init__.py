import time

import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators.operator import Operator
from fiftyone.operators.executor import ExecutionContext
import fiftyone as fo
import os
import subprocess

from neptune.legacy import session


class ClipVideo(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="clip_video",
            label="Clip Video",
            dynamic=True,
        )

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label=self.config.label, prompt=True),
        )

    def resolve_input(self, ctx: ExecutionContext):
        inputs = types.Object()
        inputs.view_target(ctx)
        inputs.int("frame_index", label="Frame Index", default=0)
        inputs.str("output_dir", label="Output Directory", default="~/clipped_videos")
        inputs.list(
            "clip_tags",
            label="Clip Tags to add.",
            element_type=types.String(),
            default=["clipped"],
        )
        return types.Property(inputs, view=types.View(label="Clip Video"))

    def execute(self, ctx: ExecutionContext):
        frame_index = ctx.params["frame_index"]
        output_dir = os.path.expanduser(ctx.params["output_dir"])
        os.makedirs(output_dir, exist_ok=True)

        for sample in ctx.target_view():
            video_path = sample.filepath
            output_path = os.path.join(output_dir, f"clipped_{os.path.basename(video_path)}")
            self.clip_video(video_path, frame_index, output_path)
            self.load_clipped_video(ctx, output_path)

        return {"output_dir": output_dir}

    def clip_video(self, video_path, frame_index, output_path):
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select='eq(n\,{frame_index})'",
            "-vsync", "vfr",
            output_path
        ]
        subprocess.run(command, check=True)
    def load_clipped_video(self, ctx, video_path):
        dataset = fo.Dataset.from_dir(
            dataset_dir=os.path.dirname(video_path),
            dataset_type=fo.types.VideoDirectory
        )
        dst_dataset = ctx.params.get("dst_dataset", None)
        dst_dataset.add_samples(dataset)
        while not os.path.exists(video_path):
            time.sleep(1)
        ctx.ops.reload_dataset()


    def resolve_output(self, ctx: ExecutionContext):
        outputs = types.Object()
        outputs.str("output_dir", label="Output Directory")
        return types.Property(outputs)

def register(p: Operator):
    p.register(ClipVideo)