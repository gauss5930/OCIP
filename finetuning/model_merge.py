from utils import (
    get_merged_model
)
import argparse
import huggingface_hub

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="The model path where adapter model was saved")
    parser.add_argument("--output_dir", type=str, help="The output directory to save the merged model")
    parser.add_argument("--hf_hub_path", type=str, help="The HF hub path to save the merged model")
    parser.add_argument("--hf_token", type=str, help="It needs to save the merged model to the HF hub")

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse

    if args.hf_token:
        huggingface_hub.login(args.hf_token)

    get_merged_model(args)