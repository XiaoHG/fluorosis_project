"""
Download Kaggle experiment results — two methods.

Method 1 (in Kaggle notebook): zip and create download link
Method 2 (local terminal): kaggle API
"""

# ============================================================
# Method 1: Run this cell in your Kaggle notebook
# ============================================================
# import os, shutil
# from IPython.display import FileLink
#
# os.makedirs("/kaggle/working/download", exist_ok=True)
#
# # Collect all result files
# files_to_download = []
# for f in os.listdir("/kaggle/working"):
#     if f.endswith(".json") or f.endswith(".pt") or f.endswith(".csv"):
#         src = os.path.join("/kaggle/working", f)
#         dst = os.path.join("/kaggle/working/download", f)
#         if os.path.isfile(src):
#             shutil.copy2(src, dst)
#             files_to_download.append(f)
#
# # Also grab model checkpoints
# for subdir in os.listdir("/kaggle/working"):
#     subpath = os.path.join("/kaggle/working", subdir)
#     if os.path.isdir(subpath) and subdir.startswith("df_"):
#         best_pt = os.path.join(subpath, "best.pt")
#         if os.path.exists(best_pt):
#             dst = os.path.join("/kaggle/working/download", f"{subdir}_best.pt")
#             shutil.copy2(best_pt, dst)
#             files_to_download.append(f"{subdir}_best.pt")
#
# print(f"Collected {len(files_to_download)} files:")
# for f in files_to_download:
#     size_kb = os.path.getsize(os.path.join("/kaggle/working/download", f)) / 1024
#     print(f"  {f} ({size_kb:.1f} KB)")
#
# # Zip
# zip_path = "/kaggle/working/fluorosis_results.zip"
# shutil.make_archive(zip_path.replace(".zip", ""), "zip", "/kaggle/working/download")
# zip_size = os.path.getsize(zip_path) / 1024 / 1024
# print(f"\nZip: {zip_path} ({zip_size:.1f} MB)")
# print("Click the file link below OR use the Kaggle output tab to download:")
# FileLink(zip_path)


# ============================================================
# Method 2: From local terminal (after Kaggle saves output)
# ============================================================
# Step 1: On Kaggle, click "Save Version" then go to "Output" tab and download
#   OR use the Kaggle API:
#
#   kaggle kernels output <kaggle-username>/<kernel-slug> -p ./downloads/
#
# Step 2: Extract
#   unzip ./downloads/fluorosis_results.zip -d ./results/
