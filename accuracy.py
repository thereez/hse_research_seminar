from argparse import ArgumentParser
import jsonlines
from datasets import load_dataset


def main():
    ds = load_dataset("parquet", data_files={"test": args.gt_path})
    with jsonlines.open(args.pred_path, 'r') as f:
        preds = list(f)

    results = []
    for gt in ds['test']:
        for pred in preds:
            if pred['id'] == gt['id']:
                results.append(int(pred['answer'] == gt['answerKey']))

    print(sum(results)/len(results))


if __name__ == "__main__":
    parser.add_argument('--gt-path', type=str)
    parser.add_argument('--pred-path', type=str)
    args = parser.parse_args()
    main(args)
