"""Unit tests for unified text pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_pipeline import TextPipeline


def test_pipeline_returns_dict():
    pipeline = TextPipeline()
    result = pipeline.process("My payment failed")
    assert isinstance(result, dict)


def test_pipeline_has_required_keys():
    pipeline = TextPipeline()
    result = pipeline.process("My payment failed")
    keys = ["original", "cleaned", "normalized", "sentiment", "keywords"]
    for key in keys:
        assert key in result


def test_pipeline_cleans_text():
    pipeline = TextPipeline()
    result = pipeline.process("My PAYMENT Failed!!!")
    assert result["cleaned"] == result["cleaned"].lower()


def test_pipeline_extracts_keywords():
    pipeline = TextPipeline()
    result = pipeline.process("My payment failed money deducted")
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) > 0


def test_pipeline_handles_empty():
    pipeline = TextPipeline()
    result = pipeline.process("")
    assert result["cleaned"] == ""


def test_pipeline_batch():
    pipeline = TextPipeline()
    texts = ["payment failed", "account locked", "need refund"]
    results = pipeline.process_batch(texts)
    assert len(results) == 3
    assert all("cleaned" in r for r in results)
