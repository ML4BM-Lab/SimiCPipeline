"""Tests for simicpipeline.utils.io utilities."""

import pickle
from pathlib import Path

import pytest

from simicpipeline.utils.io import format_time, print_tree, write_pickle


class TestFormatTime:
    """Tests for the format_time utility function."""

    def test_seconds_only(self):
        assert format_time(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_time(90) == "1min 30s"

    def test_hours_minutes_seconds(self):
        assert format_time(3661) == "1h 1min 1s"

    def test_zero(self):
        assert format_time(0) == "0s"

    def test_returns_string(self):
        assert isinstance(format_time(100), str)


class TestWritePickle:
    """Tests for the write_pickle utility function."""

    def test_writes_and_reloads(self, tmp_path):
        obj = {"key": [1, 2, 3]}
        out = tmp_path / "test.pkl"
        write_pickle(obj, out)
        assert out.exists()
        with open(out, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == obj

    def test_creates_parent_directories(self, tmp_path):
        obj = "hello"
        out = tmp_path / "nested" / "dir" / "test.pkl"
        write_pickle(obj, out)
        assert out.exists()


class TestPrintTree:
    """Tests for the print_tree utility function."""

    def test_runs_without_error(self, tmp_path, capsys):
        (tmp_path / "a.txt").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("y")
        print_tree(tmp_path, max_depth=2)
        captured = capsys.readouterr()
        assert tmp_path.name in captured.out
