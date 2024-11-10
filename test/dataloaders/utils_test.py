import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.dataloaders.utils import csv_to_jsonl, prepare_jsonl, save_as_jsonl

INPUT_CSV = "furry_50k_4o/filtered-posts-2024-06-16_with_4o_captions.csv"
OUTPUT_JSONL = "test_raw.jsonl"

csv_to_jsonl(INPUT_CSV, OUTPUT_JSONL, 1000)

jsonl_out = prepare_jsonl(
    OUTPUT_JSONL,
    filename_col="md5",
    caption_or_tags_col="tag_string",
    width_col="image_width",
    height_col="image_height",
    ext_col="file_ext",
)

save_as_jsonl(jsonl_out, "test_raw_data.jsonl")

print()
