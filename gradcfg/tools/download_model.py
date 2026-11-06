"""Download weights explicitly; importing this module has no side effects."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="segmind/tiny-sd")
    parser.add_argument("--output", default="sd_model/tiny-sd")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="Can be set to https://hf-mirror.com")
    parser.add_argument("--revision", default=None, help="Optional immutable model revision")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=args.model_id, local_dir=args.output, endpoint=args.endpoint, revision=args.revision)


if __name__ == "__main__":
    main()
