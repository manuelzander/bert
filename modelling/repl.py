import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer

from run_squad import to_list
from utils_squad import convert_examples_to_features, SquadExample, RawResult, write_predictions


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


def get_features(examples, tokenizer):
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

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids,
        all_example_index, all_cls_index, all_p_mask
    )

    return dataset, examples, features


def ask(model, tokenizer, question, context, device="cpu"):
    # create a squad example
    examples = SquadExample(
        qas_id=1000000000,
        question_text=question,
        doc_tokens=context.split(" ")
    )

    # # Make features TODO clean me up
    dataset, examples, features = get_features([examples], tokenizer)

    dataloader = DataLoader(dataset)

    # Run the model
    model.eval()  # signal eval mode
    batch = dataloader.__iter__().__next__()

    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2]  # XLM don't use segment_ids
        }
        outputs = model(**inputs)

    results = [
        RawResult(
            unique_id=1_000_000_000,
            start_logits=to_list(outputs[0][0]),
            end_logits=to_list(outputs[1][0])
        )
    ]

    preds = write_predictions(
        examples, features, results,
        1, 30, True,
        "outputs.json",
        "outputs_nbest.json",
        "output_null_log_odds.json",
        True, False, 0.0
    )

    ans = preds[1_000_000_000]
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--context")
    args = parser.parse_args()

    # load the model and the tokenizer
    model, tokenizer = get_model("/home/patrick/projects/hackathon/berthachathon/modelling")

    answer = ask(model, tokenizer, args.question, args.context)
    print(answer)
