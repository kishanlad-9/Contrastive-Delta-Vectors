import os
import sys
import shutil
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Download Fashion-IQ clothing images.")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["dress", "shirt", "toptee", "all"],
        help="Category to download: 'dress', 'shirt', 'toptee', or 'all' (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        help="Root directory where images will be saved (default: 'images')",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of concurrent download threads (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to download per category (e.g. 500 for testing, default: all)",
    )
    return parser.parse_args()


def download_single_image(asin, url, out_path, broken_links_dir):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skipped"

    # Check if there is an offline backup in broken_links
    if broken_links_dir and os.path.isdir(broken_links_dir):
        backup_file = os.path.join(broken_links_dir, f"{asin}.jpg")
        if os.path.exists(backup_file):
            shutil.copyfile(backup_file, out_path)
            return "success"

    # Download from URL
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                with open(out_path, "wb") as f:
                    f.write(response.read())
                return "success"
    except Exception:
        # Check broken links directory as fallback
        if broken_links_dir and os.path.isdir(broken_links_dir):
            backup_file = os.path.join(broken_links_dir, f"{asin}.jpg")
            if os.path.exists(backup_file):
                shutil.copyfile(backup_file, out_path)
                return "success"
        return "failed"

    return "failed"


def process_category(category, base_dir, output_root, max_workers=16, limit=None):
    txt_file = os.path.join(base_dir, "fashion-iq-metadata", "image_url", f"asin2url.{category}.txt")
    broken_dir = os.path.join(base_dir, "fashion-iq-metadata", "image_url", "broken_links")

    if not os.path.exists(txt_file):
        print(f"[-] Error: Mapping file not found: {txt_file}")
        return

    cat_output_dir = os.path.join(output_root, category)
    os.makedirs(cat_output_dir, exist_ok=True)

    # Read pairs
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                asin = parts[0]
                url = parts[1]
                items.append((asin, url))

    total_available = len(items)
    if limit is not None and limit > 0:
        items = items[:limit]

    print(f"\n==========================================")
    print(f" Category: {category.upper()}")
    print(f" Total to download: {len(items)} (out of {total_available})")
    print(f" Target folder: {cat_output_dir}")
    print(f"==========================================")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single_image,
                asin,
                url,
                os.path.join(cat_output_dir, f"{asin}.jpg"),
                broken_dir,
            ): asin
            for asin, url in items
        }

        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res == "success":
                success_count += 1
            elif res == "skipped":
                skipped_count += 1
            else:
                failed_count += 1

            if i % 100 == 0 or i == total:
                print(
                    f"[{category}] Processed {i}/{total} | "
                    f"Downloaded: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}",
                    end="\r",
                    flush=True,
                )

    print()
    print(f" Finished {category}: {success_count} downloaded, {skipped_count} already existed, {failed_count} failed.\n")


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    categories = ["dress", "shirt", "toptee"] if args.category == "all" else [args.category]

    print("Fashion-IQ Image Downloader")
    print(f"Selected categories: {categories}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Threads: {args.max_workers}")

    for cat in categories:
        process_category(
            category=cat,
            base_dir=base_dir,
            output_root=args.output_dir,
            max_workers=args.max_workers,
            limit=args.limit,
        )

    print("All tasks completed.")

if __name__ == "__main__":
    main()
