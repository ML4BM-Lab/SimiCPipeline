"""Smoke tests for core base class."""

from pathlib import Path

import pytest

from simicpipeline.core.base import SimiCBase, _load_assignment_file


class TestSimiCBase:
    """Smoke tests for SimiCBase initialization and helpers."""

    def test_creates_project_dir(self, tmp_path):
        project_dir = tmp_path / "my_project"
        assert not project_dir.exists()
        base = SimiCBase(project_dir)
        assert project_dir.exists()
        assert base.project_dir == project_dir

    def test_existing_project_dir(self, tmp_path):
        base = SimiCBase(tmp_path)
        assert base.project_dir == tmp_path

    def test_format_time_delegates(self, tmp_path):
        base = SimiCBase(tmp_path)
        assert base.format_time(60) == "1min 0s"

    def test_print_project_info(self, tmp_path, capsys):
        base = SimiCBase(tmp_path)
        base.print_project_info(max_depth=1)
        captured = capsys.readouterr()
        assert tmp_path.name in captured.out


class TestLoadAssignmentFile:
    """Tests for _load_assignment_file helper."""

    def test_new_csv_format(self, tmp_path):
        csv_file = tmp_path / "assignment.csv"
        csv_file.write_text("category,label\ncell1,0\ncell2,1\ncell3,0\n")
        labels = _load_assignment_file(csv_file)
        assert list(labels) == [0, 1, 0]

    def test_legacy_format(self, tmp_path):
        tsv_file = tmp_path / "assignment.txt"
        tsv_file.write_text("0\n1\n1\n0\n")
        labels = _load_assignment_file(tsv_file)
        assert list(labels) == [0, 1, 1, 0]
