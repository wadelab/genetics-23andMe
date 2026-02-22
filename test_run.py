#!/usr/bin/env python3
"""Quick functional test of the GUI components."""
import sys
import time
from genetics_23andme.gui import load_snps, compute_summary, find_fmf_matches

try:
    # Test 1: Load SNPs
    start = time.time()
    snps = load_snps('/Users/alexwade/Documents/genome_R3154.txt')
    load_time = time.time() - start
    print(f'✅ load_snps: {len(snps)} SNPs in {load_time:.2f}s')
    
    # Test 2: Compute summary
    summary = compute_summary(snps)
    print(f'✅ compute_summary: {summary["total_snps"]} total SNPs, {summary["rs_cnt"]} rs-prefixed')
    
    # Test 3: Find FMF matches
    fmf = find_fmf_matches(snps)
    print(f'✅ find_fmf_matches: {len(fmf)} matches found')
    
    # Test 4: Verify key data structures
    print(f'✅ Genotype counts: {summary["genotype_counts"][:3]}')
    print(f'✅ Chromosome counts: {summary["chromosome_counts"][:3]}')
    
    print('\n🎉 All core functions working!')
    sys.exit(0)
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
