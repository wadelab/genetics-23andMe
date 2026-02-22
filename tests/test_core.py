"""Tests for genetics_23andme.gui module."""

from genetics_23andme.gui import compute_summary, find_fmf_matches
from genetics_23andme.fitness import FMF_NOTEWORTHY


class TestComputeSummary:
    """Test summary computation from SNP data."""

    def test_empty_snps(self):
        """Test summary computation with no SNPs."""
        snps = {}
        summary = compute_summary(snps)

        assert summary["total_snps"] == 0
        assert summary["rs_cnt"] == 0
        assert summary["internal_cnt"] == 0

    def test_basic_snps(self):
        """Test summary computation with sample SNPs."""
        snps = {
            "rs429358": ["19", "44908822", "CT"],
            "rs7412": ["19", "44909393", "CC"],
            "i5000001": ["1", "1000000", "AA"],
        }
        summary = compute_summary(snps)

        assert summary["total_snps"] == 3
        assert summary["rs_cnt"] == 2
        assert summary["internal_cnt"] == 1
        assert summary["unique_genotypes"] == 3
        assert summary["unique_chromosomes"] == 2


def test_fmf_noteworthy_contains_data():
    """Test that FMF_NOTEWORTHY has expected data."""
    assert len(FMF_NOTEWORTHY) > 0
    assert "rs429358" in FMF_NOTEWORTHY
    assert isinstance(FMF_NOTEWORTHY["rs429358"], list)
    assert len(FMF_NOTEWORTHY["rs429358"]) == 3
