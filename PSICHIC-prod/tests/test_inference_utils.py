"""Unit tests for inference.py utility functions.

Heavy deps (torch_geometric, sklearn, etc.) are stubbed in conftest.py.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

# Add PSICHIC-prod and repo root to path
PROD_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROD_DIR.parent
for p in (str(PROD_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import (
    _build_canonical_map,
    _safe_protein_id,
    _smiles_hash,
    parse_fasta,
    parse_smiles,
    sort_csv_by_score,
)


# ---------------------------------------------------------------------------
# parse_fasta
# ---------------------------------------------------------------------------


class TestParseFasta:
    def test_single_sequence(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">protein_1\nMKTAYIAKQ\nNAPQHI\n")
        result = parse_fasta(fasta)
        assert result == {"protein_1": "MKTAYIAKQNAPQHI"}

    def test_multiple_sequences(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">P1\nAAA\n>P2\nCCC\n>P3\nDDD\n")
        result = parse_fasta(fasta)
        assert len(result) == 3
        assert result["P1"] == "AAA"
        assert result["P2"] == "CCC"
        assert result["P3"] == "DDD"

    def test_multiline_sequence(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">long_prot\nAAAA\nBBBB\nCCCC\n")
        result = parse_fasta(fasta)
        assert result["long_prot"] == "AAAABBBBCCCC"

    def test_header_with_description(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|Q16566|KCC4_HUMAN CaMK4 description\nMKT\n")
        result = parse_fasta(fasta)
        assert "sp|Q16566|KCC4_HUMAN" in result

    def test_empty_file(self, tmp_path: Path) -> None:
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        result = parse_fasta(fasta)
        assert result == {}

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">P1\n\nAAA\n\n>P2\nBBB\n")
        result = parse_fasta(fasta)
        assert result == {"P1": "AAA", "P2": "BBB"}


# ---------------------------------------------------------------------------
# parse_smiles (CSV and SMI formats)
# ---------------------------------------------------------------------------


class TestParseSmiles:
    def test_csv_with_smiles_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ligands.csv"
        csv_path.write_text("ID,SMILES\nh1,CCO\nh2,CCN\nh3,CCO\n")
        result = parse_smiles(csv_path)
        assert "CCO" in result
        assert "CCN" in result
        # Deduplication: CCO appears twice but should be unique
        assert len(result) == 2

    def test_smi_format(self, tmp_path: Path) -> None:
        smi_path = tmp_path / "ligands.smi"
        smi_path.write_text("CCO ethanol\nCCN ethylamine\n# comment\n\nCCO duplicate\n")
        result = parse_smiles(smi_path)
        assert len(result) == 2  # CCO deduped

    def test_csv_case_insensitive_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ligands.csv"
        csv_path.write_text("id,Smiles\n1,CCO\n2,CCN\n")
        result = parse_smiles(csv_path)
        assert len(result) == 2

    def test_csv_falls_back_to_first_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ligands.csv"
        csv_path.write_text("compound,activity\nCCO,5.0\nCCN,6.0\n")
        result = parse_smiles(csv_path)
        assert "CCO" in result


# ---------------------------------------------------------------------------
# _build_canonical_map
# ---------------------------------------------------------------------------


class TestBuildCanonicalMap:
    def test_valid_smiles(self) -> None:
        canonical_map, failed = _build_canonical_map(["CCO", "OCC", "C(C)O"])
        # All three are ethanol — should map to same canonical form
        assert len(failed) == 0
        canonical_values = set(canonical_map.values())
        assert len(canonical_values) == 1  # all map to same canonical

    def test_invalid_smiles_reported(self) -> None:
        canonical_map, failed = _build_canonical_map(["CCO", "INVALID_SMILES_XYZ"])
        assert "CCO" in canonical_map
        assert "INVALID_SMILES_XYZ" in failed

    def test_empty_input(self) -> None:
        canonical_map, failed = _build_canonical_map([])
        assert canonical_map == {}
        assert failed == []


# ---------------------------------------------------------------------------
# _safe_protein_id
# ---------------------------------------------------------------------------


class TestSafeProteinId:
    def test_normal_id(self) -> None:
        assert _safe_protein_id("CaMK4") == "CaMK4"

    def test_special_chars_replaced(self) -> None:
        result = _safe_protein_id("sp|Q16566|KCC4_HUMAN")
        assert "/" not in result
        assert "\\" not in result
        assert "|" not in result

    def test_dotdot_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_protein_id("../etc/passwd")

    def test_dot_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_protein_id(".")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_protein_id("")


# ---------------------------------------------------------------------------
# _smiles_hash
# ---------------------------------------------------------------------------


class TestSmilesHash:
    def test_returns_sha256_hex(self) -> None:
        result = _smiles_hash("CCO")
        expected = hashlib.sha256(b"CCO").hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_deterministic(self) -> None:
        assert _smiles_hash("C1=CC=CC=C1") == _smiles_hash("C1=CC=CC=C1")

    def test_different_smiles_different_hash(self) -> None:
        assert _smiles_hash("CCO") != _smiles_hash("CCN")

    def test_long_smiles_still_64_bytes(self) -> None:
        long_smi = "C" * 1000
        result = _smiles_hash(long_smi)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# sort_csv_by_score
# ---------------------------------------------------------------------------


class TestSortCsvByScore:
    def test_sorts_by_protein_asc_prediction_desc(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["protein_id", "smiles", "prediction", "protein_length"])
            writer.writerow(["P2", "CCO", "5.0", "100"])
            writer.writerow(["P1", "CCN", "7.0", "200"])
            writer.writerow(["P1", "CCC", "8.0", "200"])
            writer.writerow(["P2", "CCF", "6.0", "100"])

        sort_csv_by_score(csv_path)

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # P1 first (alphabetical), highest prediction first
        assert rows[0]["protein_id"] == "P1"
        assert rows[0]["prediction"] == "8.0"
        assert rows[1]["protein_id"] == "P1"
        assert rows[1]["prediction"] == "7.0"
        # P2 next
        assert rows[2]["protein_id"] == "P2"
        assert rows[2]["prediction"] == "6.0"
        assert rows[3]["protein_id"] == "P2"
        assert rows[3]["prediction"] == "5.0"


# ---------------------------------------------------------------------------
# LMDB round-trip (write → read)
# ---------------------------------------------------------------------------


class TestLmdbRoundTrip:
    def test_put_and_get(self, tmp_path: Path) -> None:
        lmdb = pytest.importorskip("lmdb")  # noqa: F841

        from inference import _lmdb_put_batch, _open_lmdb

        lmdb_path = tmp_path / "test.lmdb"
        env = _open_lmdb(lmdb_path)

        items = [
            (b"key1", b"value1"),
            (b"key2", b"value2"),
            (b"key3", b"value3"),
        ]
        skipped = _lmdb_put_batch(env, items)
        assert skipped == 0

        # Read back
        with env.begin() as txn:
            assert txn.get(b"key1") == b"value1"
            assert txn.get(b"key2") == b"value2"
            assert txn.get(b"key3") == b"value3"
            assert txn.get(b"missing") is None

        env.close()

    def test_oversize_key_skipped(self, tmp_path: Path) -> None:
        lmdb = pytest.importorskip("lmdb")  # noqa: F841

        from inference import _lmdb_put_batch, _open_lmdb

        lmdb_path = tmp_path / "test.lmdb"
        env = _open_lmdb(lmdb_path)

        # 512-byte key exceeds LMDB default max (511)
        oversize_key = b"X" * 512
        items = [
            (b"good_key", b"good_value"),
            (oversize_key, b"bad_value"),
        ]
        skipped = _lmdb_put_batch(env, items)
        assert skipped == 1

        # Good key should still be written
        with env.begin() as txn:
            assert txn.get(b"good_key") == b"good_value"
            assert txn.get(oversize_key) is None

        env.close()

    def test_hashed_keys_always_fit(self, tmp_path: Path) -> None:
        """Verify that SHA-256 hashed SMILES keys always fit in LMDB."""
        long_smiles = "C" * 1000  # Exceeds 511 bytes raw
        hashed_key = _smiles_hash(long_smiles).encode("utf-8")
        assert len(hashed_key) == 64  # Always fits
