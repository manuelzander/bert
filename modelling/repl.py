import argparse
import torch
from typing import List
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer

from run_squad import to_list
from utils_squad import convert_examples_to_features, SquadExample, RawResult, write_predictions

MODEL_NAME = "bert-base-cased"


def get_model(model_name):
    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=True
    )
    model = BertForQuestionAnswering.from_pretrained(
        model_name,
        from_tf=bool('.ckpt' in model_name),
        config=config
    )

    return model, tokenizer


def get_features(examples: List[SquadExample], tokenizer):
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask)

    return dataset, examples, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--context")
    parser.add_argument("--model")
    args = parser.parse_args()

    # load the model
    model, tokenizer = get_model(args.model)
    import pdb; pdb.set_trace()
    # create a squad example
    examples = [
        SquadExample(
            qas_id="repl",
            question_text=args.question,
            doc_tokens=args.context.split(" ")
        )
    ]
    print(examples)

    dataset, examples, features = get_features(examples, tokenizer)
    dataloader = DataLoader(dataset)

    all_results = []
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to("cpu") for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]  # XLM don't use segment_ids
            }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    preds = write_predictions(
        examples, features, all_results,
        1,
        30,
        True,
        "outputs.json",
        "outputs_nbest.json",
        "output_null_log_odds.json",
        True,
        False,
        0.0
    )

    print(preds)


if __name__ == "__main__":
    main()
