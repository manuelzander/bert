import argparse
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer

from source.config import ROOT_DIR, MODEL_NAME
from source.modelling.run_squad import to_list
from source.modelling.utils_squad import convert_examples_to_features, SquadExample, RawResult, write_predictions


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
        qas_id=1_000_000_000,
        question_text=question,
        doc_tokens=context.split(" ")
    )

    # Make features TODO clean me up
    dataset, examples, features = get_features([examples], tokenizer)
    dataloader = DataLoader(dataset)

    all_results = []
    for batch in dataloader:
        # Run the model
        model.eval()  # signal eval mode
        batch = tuple(t.to(device) for t in batch)

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
        n_best_size=1,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file="outputs.json",
        output_nbest_file="outputs_nbest.json",
        output_null_log_odds_file="output_null_log_odds.json",
        verbose_logging=True,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0
    )

    ans = preds[1_000_000_000]
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--context")
    args = parser.parse_args()

    # load the model and the tokenizer: change to the location of modelling folder
    model_path = os.path.join(ROOT_DIR, MODEL_NAME)
    model, tokenizer = get_model(model_path)

    answer = ask(model, tokenizer, args.question, args.context)
    print(answer)
