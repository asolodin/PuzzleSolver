import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="openai",  # or "portkey", etc.
    backend_kwargs={
        "model_name": "gpt-5.2",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    environment="docker",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

# get a list of files in experiments/moscow_puzzles/questions
files = os.listdir("experiments/moscow_puzzles/questions")
for file in files:
    with open(f"experiments/moscow_puzzles/questions/{file}", "r") as f:
        question = f.read()
        print(f"Question from {file}: {question}")
    result = rlm.completion(question)
    print(f"\tAnswer: {result}")
