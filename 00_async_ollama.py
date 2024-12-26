import glob
import json
import logging
import os
import re
import time
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any

import asyncio
import pandas as pd
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

# Configuration Settings
class Config:
    """Configuration settings for the testing harness."""
    MAX_RETRIES: int = 4
    RATE_LIMIT_DELAY: float = 1.0  # Seconds
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 5
    REQUEST_TIMEOUT: float = 30.0  # Seconds
    RETRY_DELAY: float = 2.0  # Seconds
    CSV_PATTERN: str = "*.csv"
    CONSOLE: Console = Console()

# Enumerations
class AnswerChoice(str, Enum):
    """Enumeration of possible answer choices."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

# Data Models
class QuestionOptions(BaseModel):
    """Model for question options and correct answer."""
    model_config = ConfigDict(protected_namespaces=())
    options: List[Tuple[AnswerChoice, str]]
    correct_letter: AnswerChoice
    letter_option_map: Dict[AnswerChoice, str]

class ModelResponse(BaseModel):
    """Model for the language model's response."""
    answer: AnswerChoice = Field(..., description="The model's answer (A, B, C, D, or E)")

    @classmethod
    def validate_answer(cls, v: Any) -> 'ModelResponse':
        if not isinstance(v, AnswerChoice):
            raise ValueError(f"Invalid answer choice: {v}")
        return cls(answer=v)

class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    model_config = ConfigDict(protected_namespaces=())
    question_number: int
    question: str
    options: QuestionOptions
    model_answer: Optional[ModelResponse] = None
    is_correct: bool = False
    error: Optional[str] = None
    duration: Optional[float] = None
    retries: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_number": self.question_number,
            "question": self.question,
            "options": self.options.dict(),
            "model_answer": self.model_answer.dict() if self.model_answer else None,
            "is_correct": self.is_correct,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
            "timestamp": self.timestamp.isoformat(),
        }

class TestCase(BaseModel):
    """Model for test cases."""
    model_config = ConfigDict(protected_namespaces=())
    id: str
    type: str
    input: str
    expected_output: str
    actual_output: Optional[str] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration: Optional[float] = None
    retries: int = 0
    error_messages: List[str] = Field(default_factory=list)

    @classmethod
    def from_evaluation_result(cls, result: EvaluationResult) -> 'TestCase':
        return cls(
            id=f"q{result.question_number}",
            type="multiple_choice",
            input=result.question,
            expected_output=result.options.correct_letter.value,
            actual_output=result.model_answer.answer.value if result.model_answer else None,
            passed=result.is_correct,
            metadata={
                "options": result.options.letter_option_map,
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            },
            duration=result.duration,
            retries=result.retries,
            error_messages=[result.error] if result.error else []
        )

class TestSuite(BaseModel):
    """Model for test suites."""
    model_config = ConfigDict(protected_namespaces=())
    name: str
    description: str
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_analytics(self) -> Dict[str, Any]:
        total = len(self.test_cases)
        passed = sum(1 for tc in self.test_cases if tc.passed)
        failed = total - passed
        average_duration = sum(tc.duration or 0 for tc in self.test_cases) / total if total > 0 else 0
        total_retries = sum(tc.retries for tc in self.test_cases)
        error_rate = (len([tc for tc in self.test_cases if tc.error_messages]) / total * 100) if total > 0 else 0

        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "average_duration": average_duration,
            "total_retries": total_retries,
            "error_rate": error_rate
        }

# Language Model Interfaces
class AsyncLanguageModel(ABC):
    """Abstract base class for async language models."""

    @abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def aclose(self):
        pass

class AsyncOllamaLanguageModel(AsyncLanguageModel):
    """Async implementation for Ollama language model."""

    def __init__(self, model: str, base_url: str, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm: Optional[OllamaLLM] = None

    async def ainvoke(self, prompt: str) -> str:
        if not self._llm:
            self._llm = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                **self.kwargs
            )
        return await asyncio.to_thread(self._llm.invoke, prompt)

    async def aclose(self):
        self._llm = None

# Utility Functions
def parse_csv_row(row: pd.Series) -> Tuple[int, str, List[str], str]:
    """Parse a row from the CSV file."""
    try:
        question_number = int(row['id'])
    except ValueError:
        raise ValueError(f"Invalid question number: {row['id']}")

    question = str(row['prompt'])
    
    # Get options A through E from the direct column names
    options = [
        str(row.get(letter, "")).strip() 
        for letter in ['A', 'B', 'C', 'D', 'E']
    ]
    
    # Remove any empty options
    options = [opt for opt in options if opt]
    
    correct_letter = str(row['answer']).strip().upper()
    
    if correct_letter not in AnswerChoice.__members__:
        raise ValueError(f"Invalid correct answer letter: {correct_letter}")
    
    return question_number, question, options, correct_letter

def prepare_options(question_data: Tuple[int, str, List[str], str]) -> QuestionOptions:
    """Prepare question options."""
    question_number, _, answer_choices, correct_letter = question_data
    # Create options with available choices
    letters = [letter.value for letter in AnswerChoice][:len(answer_choices)]
    options = list(zip(letters, answer_choices))
    letter_option_map = {AnswerChoice(letter): option for letter, option in options}

    return QuestionOptions(
        options=[(AnswerChoice(letter), option) for letter, option in options],
        correct_letter=AnswerChoice(correct_letter),
        letter_option_map=letter_option_map
    )

def parse_model_response(response: str) -> ModelResponse:
    """Parse the model's response."""
    # Pattern to match 'Answer: X' where X is A-E
    match = re.search(r"Answer\s*[:\-is]+\s*([A-E])\b", response, re.IGNORECASE | re.MULTILINE)
    
    if match:
        answer = match.group(1).upper()
        return ModelResponse(answer=AnswerChoice(answer))
    
    # Fallback: find any standalone A-E letter
    fallback_match = re.search(r"\b([A-E])\b", response, re.IGNORECASE)
    if fallback_match:
        answer = fallback_match.group(1).upper()
        logging.debug(f"Fallback parsing successful: Found '{answer}' in response.")
        return ModelResponse(answer=AnswerChoice(answer))
    
    logging.debug(f"Unparsable model response: '{response}'")
    raise ValueError("Could not parse model response")

# Main Test Harness Class
class TestHarness:
    """Main test harness class."""

    def __init__(self, model: AsyncLanguageModel, output_dir: str, verbose: bool = False):
        self.model = model
        self.output_dir = output_dir
        self.verbose = verbose
        self.console = Config.CONSOLE
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = os.path.join(
            self.output_dir,
            f"test_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    async def evaluate_question(
        self,
        row: pd.Series,
        question_number: int,
        retry_count: int = 0
    ) -> EvaluationResult:
        """Evaluate a single question."""
        start_time = time.time()
        result = EvaluationResult(
            question_number=question_number,
            question=str(row['prompt']),
            options=QuestionOptions(
                options=[],
                correct_letter=AnswerChoice.A,  # Placeholder
                letter_option_map={}
            )
        )
        
        try:
            question_data = parse_csv_row(row)
            result.options = prepare_options(question_data)
            prompt = self._create_prompt(question_data[1], result.options)
            response = await self.model.ainvoke(prompt)
            model_response = parse_model_response(response)
            result.model_answer = model_response
            result.is_correct = model_response.answer == result.options.correct_letter
            
        except Exception as e:
            if retry_count < Config.MAX_RETRIES:
                logging.warning(
                    f"Retrying question {question_number} due to error: {e}. "
                    f"Retry {retry_count + 1}/{Config.MAX_RETRIES}"
                )
                await asyncio.sleep(Config.RETRY_DELAY)
                return await self.evaluate_question(row, question_number, retry_count + 1)
            result.error = str(e)
            logging.error(f"Failed to evaluate question {question_number}: {e}")
            
        finally:
            result.duration = time.time() - start_time
            result.retries = retry_count

        return result

    async def process_question_batch(
        self,
        questions: List[Tuple[pd.Series, int]]
    ) -> List[EvaluationResult]:
        """Process a batch of questions concurrently."""
        tasks = [self.evaluate_question(row, qnum) for row, qnum in questions]
        return await asyncio.gather(*tasks)

    def _create_prompt(self, question: str, options: QuestionOptions) -> str:
        """Create the prompt for the model."""
        options_text = "\n".join(
            f"{letter.value}. {option}" for letter, option in options.options
        )
        return (
            f"Question: {question}\n"
            f"Options:\n{options_text}\n\n"
            "Please provide the answer in the following exact format and nothing else:\n"
            "Answer: <LETTER>\n"
            "Where <LETTER> is one of A, B, C, D, or E."
        )

    async def run(self, csv_dir: str) -> TestSuite:
        """Run the test harness."""
        csv_files = glob.glob(os.path.join(csv_dir, Config.CSV_PATTERN))
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_dir}")

        all_results = []
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            for csv_file in csv_files:
                task_id = progress.add_task(
                    f"Processing {os.path.basename(csv_file)}",
                    total=100
                )
                try:
                    # Read CSV with correct column names and types
                    df = pd.read_csv(
                        csv_file,
                        dtype={
                            'id': str,
                            'prompt': str,
                            'A': str,
                            'B': str,
                            'C': str,
                            'D': str,
                            'E': str,
                            'answer': str
                        }
                    )

                    # Validate DataFrame structure
                    required_columns = {'id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'}
                    missing_columns = required_columns - set(df.columns)
                    if missing_columns:
                        raise ValueError(
                            f"CSV file {csv_file} is missing required columns: {missing_columns}"
                        )

                    # Clean the data
                    df = df.dropna(subset=['id', 'prompt', 'answer'])
                    df = df.map(lambda x: str(x).strip() if isinstance(x, str) else x)

                    # Validate correct answer column
                    invalid_answers = df['answer'].apply(
                        lambda x: str(x).upper().strip() not in AnswerChoice.__members__
                    )
                    if invalid_answers.any():
                        invalid_rows = df[invalid_answers].index.tolist()
                        logging.warning(
                            f"Skipping rows {invalid_rows} in {csv_file} due to "
                            "invalid correct answer values"
                        )
                        df = df[~invalid_answers]

                    # Convert to list of tuples (row, question_number)
                    questions = [
                        (row, i) for i, row in df.iterrows()
                    ]

                    # Process in batches
                    for i in range(0, len(questions), Config.BATCH_SIZE):
                        batch = questions[i:i + Config.BATCH_SIZE]
                        results = await self.process_question_batch(batch)
                        all_results.extend(results)
                        progress.update(
                            task_id,
                            completed=(i + len(batch)) / len(questions) * 100
                        )
                        await asyncio.sleep(Config.RATE_LIMIT_DELAY)

                except Exception as e:
                    logging.error(f"Error processing file {csv_file}: {str(e)}")
                    progress.remove_task(task_id)
                    continue

        if not all_results:
            raise ValueError("No valid results were generated from any CSV files")

        # Save results to JSON file
        results_file = os.path.join(
            self.output_dir,
            f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        test_suite = self._create_test_suite(all_results)
        with open(results_file, 'w') as f:
            json.dump(test_suite.dict(), f, indent=2)

        return test_suite

    def _create_test_suite(self, results: List[EvaluationResult]) -> TestSuite:
        """Create a test suite from results."""
        test_cases = [TestCase.from_evaluation_result(result) for result in results]
        analytics = TestSuite(
            name="Multiple Choice Evaluation",
            description="Evaluation of model performance on multiple choice questions",
            test_cases=test_cases
        ).get_analytics()

        return TestSuite(
            name="Multiple Choice Evaluation",
            description="Evaluation of model performance on multiple choice questions",
            test_cases=test_cases,
            metadata={
                "timestamp": datetime.now().isoformat(),
                **analytics
            }
        )

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced testing harness for multiple choice questions."
    )
    parser.add_argument(
        "csv_dir",
        type=str,
        help="Directory containing CSV files with questions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vanilj/Phi-4:latest",
        help="Model to use for evaluation (default: vanilj/Phi-4:latest)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help=f"Batch size for processing questions (default: {Config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=Config.MAX_RETRIES,
        help=f"Maximum number of retries per question (default: {Config.MAX_RETRIES})"
    )

    args = parser.parse_args()

    # Update config based on arguments
    Config.BATCH_SIZE = args.batch_size
    Config.MAX_RETRIES = args.max_retries

    # Initialize model
    model = AsyncOllamaLanguageModel(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=10,
        stop_sequences=["\n"]
    )

    # Initialize and run test harness
    harness = TestHarness(
        model=model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    try:
        test_suite = await harness.run(args.csv_dir)
        analytics = test_suite.get_analytics()

        # Display results using Rich
        table = Table(title="Test Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in analytics.items():
            display_key = key.replace("_", " ").title()
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            table.add_row(display_key, display_value)

        Config.CONSOLE.print(table)

    finally:
        await model.aclose()

if __name__ == "__main__":
    asyncio.run(main())