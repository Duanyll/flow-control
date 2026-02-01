import io
import os
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from flow_control.utils.common import pil_to_tensor
from flow_control.utils.logging import console


class PrismLayersProDataset(Dataset):
    _last_loaded_file: str | None = None
    _last_table: pa.Table

    def __init__(self, path: str, ignore_caption: bool = False):
        self.path = path
        self.ignore_caption = ignore_caption
        index_file = os.path.join(path, "index.pkl")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"PrismLayersPro index file not found: {index_file}. "
                "Create it with the provided script."
            )

        with open(index_file, "rb") as f:
            self.index = pickle.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        record = self.index[idx]
        file = record["file"]
        row = record["row"]

        if self._last_loaded_file != file:
            filepath = os.path.join(self.path, file)
            self._last_table = pq.read_table(filepath)
            self._last_loaded_file = file

        table = self._last_table
        row_data = table.slice(row, 1).to_pylist()[0]
        for k, v in row_data.items():
            if isinstance(v, dict) and "bytes" in v:
                row_data[k] = pil_to_tensor(
                    Image.open(io.BytesIO(v["bytes"])).convert("RGBA")
                )

        h, w = row_data["whole_image"].shape[1:3]
        output = {
            "__key__": record["id"],
            "prompt": row_data["whole_caption"],
            "clean_image": row_data["whole_image"],
            "style_category": row_data["style_category"],
            "layer_images": [row_data["base_image"]],
            "layer_prompts": [row_data["base_caption"]],
            "layer_boxes": [(0, h, 0, w)],
        }
        for i in range(row_data["layer_count"]):
            prefix = f"layer_{i:02d}"
            output["layer_images"].append(row_data[f"{prefix}"])
            output["layer_prompts"].append(row_data[f"{prefix}_caption"])
            x_min, y_min, x_max, y_max = row_data[f"{prefix}_box"]
            top, bottom, left, right = y_min, y_max, x_min, x_max
            output["layer_boxes"].append((top, bottom, left, right))

        if self.ignore_caption:
            del output["prompt"]
            del output["layer_prompts"]

        return output


if __name__ == "__main__":
    import sys

    from rich.progress import track

    directory = sys.argv[1]
    files = os.listdir(directory)
    files = sorted(files)
    index = []
    for file in track(files, description="Generating index", console=console):
        if file.endswith(".parquet"):
            filepath = os.path.join(directory, file)
            table = pq.read_table(filepath, columns=["id"])
            ids = table.column("id").to_pylist()
            for i, id in enumerate(ids):
                index.append({"id": id, "file": file, "row": i})
    index_file = os.path.join(directory, "index.pkl")
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    console.print(f"Index saved to {index_file}, total {len(index)} records.")
