import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=3600.0)


def main():
    parser = argparse.ArgumentParser(description="Run a deep research query.")
    parser.add_argument(
        "--query", type=str, required=True, help="The research query to execute."
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the output file."
    )
    args = parser.parse_args()

    print("Running with:")
    print("-----")
    print(args.query)
    print("-----")
    response = client.responses.create(
        model="o3-deep-research",
        input=args.query,
        text={},
        tools=[{"type": "web_search_preview"}],
    )

    output = response.output_text

    with open(args.output_path, "w") as f:
        f.write(output)
    print(f"Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
