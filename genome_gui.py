#!/usr/bin/env python3

"""Simple GUI for analyzing 23andMe SNP data."""

import os
import csv
import re
import json
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.parse import quote
from urllib.request import urlopen, Request

from foundMyFitness import FMF_NOTEWORTHY

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
}


def load_snps(file_path):
    """Return SNP dict {rsid: [chromosome, position, genotype]} from 23andMe file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    snps = {}
    firstline = True
    with open(file_path, "r") as fp:
        for line in fp:
            items = line.split()
            if not items:
                continue

            if firstline:
                firstline = False
                if "23andMe" not in items:
                    raise ValueError("Not a 23andMe raw data file")
                continue

            if items[0].startswith("#"):
                continue

            if len(items) < 4:
                continue

            snps[items[0]] = items[1:]

    return snps


def load_snpedia_records(file_path):
    """Load SNPedia export (TSV/CSV) into list of records with magnitude sorting support."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    delimiter = "\t" if file_path.lower().endswith(".tsv") else ","
    records = []

    with open(file_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter=delimiter)
        for row in reader:
            rsid = (row.get("rsid") or row.get("RSID") or "").strip()
            if not rsid:
                continue

            try:
                magnitude = float(row.get("magnitude") or row.get("Magnitude") or 0)
            except Exception:
                magnitude = 0.0

            repute = (row.get("repute") or row.get("Repute") or "").strip()
            summary = (row.get("summary") or row.get("Summary") or row.get("description") or "").strip()

            records.append(
                {
                    "rsid": rsid,
                    "magnitude": magnitude,
                    "repute": repute,
                    "summary": summary,
                }
            )

    return records


def compute_apoe_haplotype(snps):
    """Derive APOE haplotype (ε2/ε3/ε4) from rs429358 and rs7412 genotypes.
    Returns (haplotype_str, details_str) or (None, reason).
    """
    if not snps:
        return None, "APOE not computed (no SNP data)"
    r429 = snps.get("rs429358", [None, None, None])[2]
    r7412 = snps.get("rs7412", [None, None, None])[2]
    if not r429 or not r7412 or r429 == "--" or r7412 == "--":
        return None, "APOE not available (missing rs429358/rs7412)"

    # Normalize to uppercase single-letter bases
    r429 = r429.upper()
    r7412 = r7412.upper()

    # Allowed genotype combinations mapping (unphased)
    # Reference table commonly used in genetics:
    # rs429358 TT + rs7412 CC -> ε3/ε3
    # rs429358 TT + rs7412 CT -> ε2/ε3
    # rs429358 TT + rs7412 TT -> ε2/ε2
    # rs429358 CT + rs7412 CC -> ε3/ε4
    # rs429358 CC + rs7412 CC -> ε4/ε4
    # rs429358 CT + rs7412 CT -> ε2/ε4 (only consistent heterozygote combo)
    combo = ("".join(sorted(r429)), "".join(sorted(r7412)))
    table = {
        ("TT", "CC"): "ε3/ε3",
        ("TT", "CT"): "ε2/ε3",
        ("TT", "TT"): "ε2/ε2",
        ("CT", "CC"): "ε3/ε4",
        ("CC", "CC"): "ε4/ε4",
        ("CT", "CT"): "ε2/ε4",
    }
    hap = table.get(combo)
    if not hap:
        return None, f"APOE undetermined from genotypes rs429358={r429}, rs7412={r7412}"

    details = f"APOE haplotype inferred as {hap} from rs429358={r429}, rs7412={r7412}."
    return hap, details


def synthesize_snpedia_with_llm(snpedia_matches, snp_dict=None):
    """
    Use OpenAI LLM to synthesize SNPedia results into a natural language summary.
    
    Args:
        snpedia_matches: List of dicts with rsid, magnitude, repute, summary
    
    Returns:
        Synthesized text or error message
    """
    if not HAS_OPENAI:
        return "Error: openai package not installed. Install with: pip install openai"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set"
    
    # Balance selection by repute so we include both protective and risk variants
    good = [r for r in snpedia_matches if (r.get("repute", "").lower() == "good")]
    bad = [r for r in snpedia_matches if (r.get("repute", "").lower() == "bad")]
    unknown = [r for r in snpedia_matches if r not in good + bad]

    good_sorted = sorted(good, key=lambda x: x.get("magnitude", 0), reverse=True)[:8]
    bad_sorted = sorted(bad, key=lambda x: x.get("magnitude", 0), reverse=True)[:8]
    # If one side has fewer, top up with unknowns by magnitude
    needed = 16 - (len(good_sorted) + len(bad_sorted))
    if needed > 0 and unknown:
        unknown_sorted = sorted(unknown, key=lambda x: x.get("magnitude", 0), reverse=True)[:needed]
        balanced_snps = good_sorted + bad_sorted + unknown_sorted
    else:
        balanced_snps = good_sorted + bad_sorted

    # Ensure APOE is represented even if it wasn't in the top/matched set
    apoe_info = None
    if snp_dict is not None:
        hap, details = compute_apoe_haplotype(snp_dict)
        if hap:
            apoe_info = {
                "rsid": "APOE",
                "genotype": hap,
                "magnitude": 5.0,  # prioritize in synthesis prompt
                "repute": "context",
                "summary": (
                    details
                    + " See SNPedia pages: rs429358 (https://snpedia.org/index.php/rs429358) and rs7412 (https://snpedia.org/index.php/rs7412)."
                ),
            }
            # Prepend APOE info to balanced list (avoid duplicates)
            balanced_snps = [apoe_info] + balanced_snps

    # Format SNP data for LLM with more context (include genotype when available)
    snp_text = ""
    for snp in balanced_snps:
        rsid = snp.get("rsid", "unknown")
        mag = snp.get("magnitude", 0)
        repute = snp.get("repute", "")
        genotype = snp.get("genotype", "")
        summary = snp.get("summary", "")
        snp_text += f"- **{rsid}** (Genotype: {genotype}, Magnitude: {mag}, Repute: {repute}): {summary}\n"
    
    prompt = f"""Based on the following genetic SNP findings from SNPedia, provide a comprehensive, detailed analysis of what these variants suggest about this person's health, disease risks, and traits. 

IMPORTANT INSTRUCTIONS:
- Be thorough and detailed (4-5 paragraphs minimum)
- Group findings by health domains (cardiovascular, metabolic, neurological, etc.) where applicable
- Include specific risk percentages or effect sizes where mentioned in the summaries
- Discuss gene-gene interactions where relevant
- Include lifestyle/environmental factors that may interact with these variants
- For each major finding, include a reference link in the format: [SNPedia](https://snpedia.org/index.php/{rsid})
- Use markdown formatting for emphasis and clarity
- Be scientifically accurate but write for a non-specialist audience

SECTION: POSITIVE VS NEGATIVE ALLELES
- Provide a balanced analysis considering both protective ("Good" repute) and risk ("Bad" repute) alleles.
- Explicitly call out which variants appear protective vs which are risk-enhancing for key outcomes.
- Weigh the combined effect to describe whether the overall profile tilts toward greater resilience or elevated risk in major domains.

SECTION: LONGEVITY GENES
- Identify and discuss any variants related to lifespan/healthspan including APOE (ε2/ε3/ε4 via rs429358/rs7412), FOXO3 (e.g., rs2802292), CETP (e.g., rs5882), TERT/TERC, IL6, SIRT1, IGF1/IGF1R, mTOR/AKT/PI3K pathway markers, and inflammation/immune loci (e.g., HLA, CRP).
- If these are present in the list, explain their known associations (longer or shorter lifespan, exceptional longevity, cognitive aging, CVD risk), and interpret in the context of the user's genotype and repute.

SECTION: LIFESPAN & HEALTHSPAN PREDICTIONS
- Based on the aggregate genetic risk profile (considering cardiovascular, metabolic, neurological, and cancer risk variants), provide a general assessment of how these variants might influence:
  * Expected lifespan trajectory compared to population average
  * Healthspan (years of healthy, disease-free life) predictions
  * Critical decades where health interventions would be most impactful
- Frame this as statistical associations, not certainties - genetics is ~30% of lifespan, lifestyle is ~70%
- Suggest specific preventive measures and lifestyle optimizations that could extend both lifespan and healthspan given the genetic profile
- Include discussion of the "modifiable vs non-modifiable" aspects

Top SNP findings:
{snp_text}

Please provide a detailed synthesis organized by health domains, with specific actionable insights, relevant SNPedia links, and a section on lifespan/healthspan predictions with lifestyle optimization strategies."""
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"


def compute_summary(snps):
    """Return summary counts and frequency lists for the provided SNPs."""
    rs_cnt, i_cnt = 0, 0
    genotype_counts = {}
    chromosome_counts = {}
    positions = set()

    for key, val in snps.items():
        if key.startswith("rs"):
            rs_cnt += 1
        else:
            i_cnt += 1

        chrom = val[0]
        if chrom in ("X", "Y"):
            chrom = "X?"

        chromosome_counts[chrom] = chromosome_counts.get(chrom, 0) + 1
        positions.add(val[1])

        gt = val[2]
        genotype_counts[gt] = genotype_counts.get(gt, 0) + 1

    sorted_genotypes = sorted(genotype_counts.items(), key=lambda kv: kv[1], reverse=True)
    sorted_chromosomes = sorted(chromosome_counts.items(), key=lambda kv: kv[1], reverse=True)

    summary = {
        "total_snps": len(snps),
        "rs_cnt": rs_cnt,
        "internal_cnt": i_cnt,
        "unique_positions": len(positions),
        "unique_genotypes": len(genotype_counts),
        "unique_chromosomes": len(chromosome_counts),
        "genotype_counts": sorted_genotypes,
        "chromosome_counts": sorted_chromosomes,
    }

    return summary


def complement_genotype(genotype):
    """Return the base-pair complement for a genotype like 'AG'."""
    return "".join(COMPLEMENT.get(base, base) for base in genotype)


def find_fmf_matches(snps):
    """Return list of Found My Fitness matches with match classification."""
    matches = []

    for key, val in FMF_NOTEWORTHY.items():
        raw_key = key[1:] if key.startswith("*") else key
        if raw_key not in snps:
            continue

        chrom, pos, genotype = snps[raw_key]
        fmf_genotype = val[1]

        if genotype == "--":
            match_type = "missing"
        elif genotype == fmf_genotype:
            match_type = "match"
        elif complement_genotype(fmf_genotype) == genotype:
            match_type = "complement"
        else:
            match_type = "other"

        matches.append(
            {
                "chrom": chrom,
                "rsid": raw_key,
                "genotype": genotype,
                "fmf_genotype": fmf_genotype,
                "gene": val[0],
                "note": val[2],
                "match_type": match_type,
            }
        )

    return matches


def find_snpedia_matches(snps, records):
    """Return list of SNPedia matches with magnitude and summary info."""
    matches = []

    for record in records:
        rsid = record["rsid"]
        if rsid not in snps:
            continue

        chrom, pos, genotype = snps[rsid]
        matches.append(
            {
                "chrom": chrom,
                "rsid": rsid,
                "genotype": genotype,
                "magnitude": record.get("magnitude", 0) or 0,
                "repute": record.get("repute", ""),
                "summary": record.get("summary", ""),
            }
        )

    # Sort by magnitude descending by default
    return sorted(matches, key=lambda m: m.get("magnitude", 0), reverse=True)


def fetch_snpedia_record(rsid):
    """Fetch a single SNPedia page via MediaWiki API and parse magnitude/repute/summary.

    This uses the bots.snpedia.com API (no auth). We parse wikitext for template fields
    like `| magnitude = 2.1`, `| repute = Good/Bad`, `| summary = ...`.
    """

    api = "https://bots.snpedia.com/api.php"
    query = f"{api}?action=query&prop=revisions&rvprop=content&rvslots=main&formatversion=2&format=json&titles={quote(rsid)}"

    req = Request(query, headers={"User-Agent": "genetics-23andMe/1.0"})
    with urlopen(req, timeout=4) as resp:
        payload = json.load(resp)

    pages = payload.get("query", {}).get("pages", [])
    content = ""
    if pages:
        page = pages[0]
        revisions = page.get("revisions", [])
        if revisions:
            rev0 = revisions[0]
            slots = rev0.get("slots", {})
            content = slots.get("main", {}).get("content", "") or ""
            if not content:
                # Fall back to legacy fields if present
                content = rev0.get("content", "") or rev0.get("*", "") or ""

    # Parse magnitude/repute/summary from wikitext
    mag_match = re.search(r"magnitude\s*=\s*([-+]?[0-9]+(?:\.[0-9]+)?)", content, re.IGNORECASE)
    rep_match = re.search(r"repute\s*=\s*([A-Za-z]+)", content, re.IGNORECASE)
    
    # Try multiple fields for summary with multi-line support
    sum_match = re.search(r"summary\s*=\s*(.+?)(?=\n\||" + r'"}}'+ ")", content, re.IGNORECASE | re.DOTALL)
    if not sum_match or not sum_match.group(1).strip():
        sum_match = re.search(r"stabilized\s*=\s*(.+?)(?=\n\||" + r'"}}'+ ")", content, re.IGNORECASE | re.DOTALL)
    if not sum_match or not sum_match.group(1).strip():
        # Try to get first text paragraph after the infobox
        sum_match = re.search(r"}}\s*\n+([^\n{]+)", content)

    magnitude = float(mag_match.group(1)) if mag_match else 0.0
    repute = rep_match.group(1).strip() if rep_match else ""
    summary = sum_match.group(1).strip() if sum_match else ""
    
    # Clean up wiki markup from summary
    summary = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", summary)  # [[link|text]] -> text
    summary = re.sub(r"\[\[([^\]]+)\]\]", r"\1", summary)  # [[link]] -> link
    summary = re.sub(r"'{2,}", "", summary)  # remove wiki bold/italic
    summary = re.sub(r"<ref[^>]*>.*?</ref>", "", summary, flags=re.DOTALL)  # remove refs
    summary = re.sub(r"\s+", " ", summary).strip()  # normalize whitespace

    return {
        "rsid": rsid,
        "magnitude": magnitude,
        "repute": repute,
        "summary": summary,
    }


def fetch_snpedia_batch(snp_genotype_pairs):
    """
    Fetch multiple SNPs from SNPedia in a single API request.
    For each SNP, fetch both general (rs1234) and genotype-specific (rs1234(A;G)) pages.
    
    Args:
        snp_genotype_pairs: List of tuples [(rsid, genotype), ...]
    
    Returns:
        List of records with magnitude, repute, summary
    """
    if not snp_genotype_pairs:
        return []
    
    api = "https://bots.snpedia.com/api.php"
    
    # Build list of page titles to fetch
    titles = []
    for rsid, genotype in snp_genotype_pairs:
        titles.append(rsid)  # general page
        if genotype and genotype != "--" and genotype != "II" and genotype != "DD" and genotype != "DI":
            # Format: rs1234(A;G) for genotype AG, rs1234(A) for single base
            if len(genotype) == 2:
                geno_formatted = f"{genotype[0]};{genotype[1]}"
            elif len(genotype) == 1:
                geno_formatted = genotype
            else:
                continue  # skip malformed genotypes
            titles.append(f"{rsid}({geno_formatted})")
    
    # Join titles with pipe separator
    titles_str = "|".join(titles)
    query = f"{api}?action=query&prop=revisions&rvprop=content&rvslots=main&formatversion=2&format=json&titles={quote(titles_str)}"
    
    # Fetch all pages
    req = Request(query, headers={"User-Agent": "genetics-23andMe/1.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.load(resp)
    except Exception:
        return []
    
    pages = payload.get("query", {}).get("pages", [])
    
    # Build a map of title -> content
    page_map = {}
    for page in pages:
        title = page.get("title", "")
        content = ""
        revisions = page.get("revisions", [])
        if revisions:
            rev0 = revisions[0]
            slots = rev0.get("slots", {})
            content = slots.get("main", {}).get("content", "") or ""
            if not content:
                content = rev0.get("content", "") or rev0.get("*", "") or ""
        
        # Determine if this is a general or genotype-specific page
        if "(" in title and ")" in title:
            # Genotype-specific page like rs1234(A;G)
            rsid_key = title.split("(")[0].lower()  # LOWERCASE for case-insensitive lookup
            if rsid_key not in page_map:
                page_map[rsid_key] = {}
            page_map[rsid_key]["genotype"] = content
        else:
            # General page like rs1234
            rsid_key = title.lower()  # LOWERCASE for case-insensitive lookup
            if rsid_key not in page_map:
                page_map[rsid_key] = {}
            page_map[rsid_key]["general"] = content
    
    # Parse records
    records = []
    for rsid, genotype in snp_genotype_pairs:
        try:
            rsid_lower = rsid.lower()  # LOWERCASE for lookup
            content_general = page_map.get(rsid_lower, {}).get("general", "")
            content_genotype = page_map.get(rsid_lower, {}).get("genotype", "")
            
            # Parse magnitude/repute from genotype-specific page FIRST (has the real data)
            # Then fall back to general page
            mag_match = None
            rep_match = None
            
            if content_genotype:
                mag_match = re.search(r"magnitude\s*=\s*([-+]?[0-9]+(?:\.[0-9]+)?)", content_genotype, re.IGNORECASE)
                rep_match = re.search(r"repute\s*=\s*([A-Za-z]+)", content_genotype, re.IGNORECASE)
            
            # Fall back to general page if not found in genotype
            if not mag_match and content_general:
                mag_match = re.search(r"magnitude\s*=\s*([-+]?[0-9]+(?:\.[0-9]+)?)", content_general, re.IGNORECASE)
            if not rep_match and content_general:
                rep_match = re.search(r"repute\s*=\s*([A-Za-z]+)", content_general, re.IGNORECASE)
            
            magnitude = float(mag_match.group(1)) if mag_match else 0.0
            repute = rep_match.group(1).strip() if rep_match else ""
            
            # Extract summary - prefer genotype-specific, fallback to general
            summary = ""
            
            # Try genotype-specific page first
            if content_genotype:
                # Look for any substantial text content
                # Remove the infobox template first
                clean = re.sub(r"\{\{[^}]+\}\}", "", content_genotype, count=1)
                # Get first meaningful paragraph
                paragraphs = [p.strip() for p in clean.split('\n\n') if p.strip() and not p.strip().startswith('==')]
                if paragraphs:
                    summary = paragraphs[0][:300]
                    if summary:
                        summary = f"[{genotype}] {summary}"
            
            # Fallback to general page
            if not summary and content_general:
                # Try summary field
                sum_match = re.search(r"[|]\s*summary\s*=\s*([^\n|]+)", content_general, re.IGNORECASE)
                if sum_match:
                    summary = sum_match.group(1).strip()
                
                # Try getting text after template
                if not summary:
                    clean = re.sub(r"\{\{[^}]+\}\}", "", content_general, count=1)
                    paragraphs = [p.strip() for p in clean.split('\n\n') if p.strip() and not p.strip().startswith('==')]
                    if paragraphs:
                        summary = paragraphs[0][:300]
            
            # Clean up wiki markup from summary
            if summary:
                summary = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", summary)
                summary = re.sub(r"\[\[([^\]]+)\]\]", r"\1", summary)
                summary = re.sub(r"'{2,}", "", summary)
                summary = re.sub(r"<ref[^>]*>.*?</ref>", "", summary, flags=re.DOTALL)
                summary = re.sub(r"<[^>]+>", "", summary)
                summary = re.sub(r"\s+", " ", summary).strip()
                # Truncate if too long
                if len(summary) > 300:
                    summary = summary[:297] + "..."
            
            records.append({
                "rsid": rsid,
                "magnitude": magnitude,
                "repute": repute,
                "summary": summary,
            })
        except Exception:
            # Skip SNPs that fail to parse
            records.append({
                "rsid": rsid,
                "magnitude": 0.0,
                "repute": "",
                "summary": "",
            })
    
    return records


class SNPAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧬 23andMe SNP Analyzer")
        self.geometry("1200x800")
        self.minsize(1000, 650)

        # Color scheme
        self.bg_color = "#f5f7fa"
        self.accent_color = "#4a90e2"
        self.header_bg = "#2c3e50"
        self.card_bg = "#ffffff"
        
        self.configure(bg=self.bg_color)
        self.file_path = tk.StringVar()
        self.snpedia_file = tk.StringVar()
        self.snpedia_records = []
        self.snpedia_matches = []
        self.snpedia_sort_desc = True
        self.snpedia_min_mag = 1.5  # show stronger signals first
        self.snpedia_top_n = 500  # increased from 150 since batching is fast
        self.snpedia_top_n_var = tk.IntVar(value=self.snpedia_top_n)
        self.snpedia_filter_high_mag = True
        self.snpedia_filter_var = tk.BooleanVar(value=True)
        self.snpedia_mag_threshold_var = tk.DoubleVar(value=1.5)
        self.snpedia_sample_size_var = tk.IntVar(value=1500)
        self.fetch_thread = None
        self.fetch_progress = 0
        self.fetch_total = 0
        self.current_snps = None

        # Apply modern styling
        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        """Configure modern ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main button style
        style.configure('Accent.TButton',
                       background=self.accent_color,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 10),
                       font=('Helvetica', 11, 'bold'))
        style.map('Accent.TButton',
                 background=[('active', '#3a7bc8'), ('pressed', '#2e5f9e')])
        
        # Configure secondary button style
        style.configure('Secondary.TButton',
                       background='#95a5a6',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 10),
                       font=('Helvetica', 11))
        style.map('Secondary.TButton',
                 background=[('active', '#7f8c8d'), ('pressed', '#6c7a7b')])
        
        # Configure entry style
        style.configure('Modern.TEntry',
                       fieldbackground='white',
                       borderwidth=1,
                       relief='solid')
        
        # Configure label frame
        style.configure('Card.TLabelframe',
                       background=self.card_bg,
                       borderwidth=1,
                       relief='solid')
        style.configure('Card.TLabelframe.Label',
                       background=self.card_bg,
                       foreground='#2c3e50',
                       font=('Helvetica', 12, 'bold'))
        
        # Configure treeview
        style.configure('Modern.Treeview',
                       background='white',
                       fieldbackground='white',
                       foreground='#2c3e50',
                       rowheight=28,
                       font=('Helvetica', 10))
        style.configure('Modern.Treeview.Heading',
                       background=self.accent_color,
                       foreground='white',
                       relief='flat',
                       font=('Helvetica', 10, 'bold'))
        style.map('Modern.Treeview.Heading',
                 background=[('active', '#3a7bc8')])
        style.map('Modern.Treeview',
                 background=[('selected', '#e3f2fd')],
                 foreground=[('selected', '#2c3e50')])

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Header section with dark background
        header_container = tk.Frame(self, bg=self.header_bg)
        header_container.grid(row=0, column=0, sticky="ew")
        header_container.columnconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(header_container, 
                              text="🧬 23andMe SNP Analyzer",
                              font=('Helvetica', 20, 'bold'),
                              bg=self.header_bg,
                              fg='white',
                              pady=15)
        title_label.grid(row=0, column=0, columnspan=4, sticky="ew")
        
        # File selection section
        file_frame = tk.Frame(header_container, bg=self.header_bg, pady=10)
        file_frame.grid(row=1, column=0, columnspan=5, sticky="ew", padx=20)
        file_frame.columnconfigure(1, weight=1)
        
        file_label = tk.Label(file_frame, 
                             text="📁 Genome File:",
                             font=('Helvetica', 11),
                             bg=self.header_bg,
                             fg='white')
        file_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        entry = ttk.Entry(file_frame, textvariable=self.file_path, width=80, style='Modern.TEntry')
        entry.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Button(file_frame, text="📂 Browse", command=self.open_file, style='Secondary.TButton').grid(row=0, column=2, padx=5)
        ttk.Button(file_frame, text="🔬 Analyze", command=self.analyze_file, style='Accent.TButton').grid(row=0, column=3, padx=5)
        ttk.Button(file_frame, text="📡 Fetch SNPedia", command=self.fetch_snpedia_api, style='Secondary.TButton').grid(row=0, column=4, padx=5)

        # Status bar with subtle background
        status_frame = tk.Frame(self, bg='#ecf0f1', pady=8)
        status_frame.grid(row=1, column=0, sticky="ew")
        self.status = tk.Label(status_frame,
                              text="👋 Welcome! Select a 23andMe raw data file to begin analysis.",
                              font=('Helvetica', 10),
                              bg='#ecf0f1',
                              fg='#34495e')
        self.status.pack(padx=20, anchor='w')

        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=240, maximum=1)
        self.progress.pack(padx=20, pady=(2, 4), anchor='w')

        # Main content area with padding
        content_container = tk.Frame(self, bg=self.bg_color)
        content_container.grid(row=2, column=0, sticky="nsew", padx=15, pady=15)
        content_container.columnconfigure(0, weight=1)
        content_container.rowconfigure(1, weight=3)
        
        # Summary card
        summary_frame = ttk.Labelframe(content_container, text="📊 Genomic Summary", padding=15, style='Card.TLabelframe')
        summary_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.summary_text = tk.Text(summary_frame, 
                                   height=10, 
                                   wrap="word", 
                                   state="disabled",
                                   bg='white',
                                   fg='#2c3e50',
                                   font=('Monaco', 10),
                                   relief='flat',
                                   borderwidth=0,
                                   padx=10,
                                   pady=10)
        self.summary_text.pack(fill="both", expand=True)
        
        # Configure text tags for colorful summary
        self.summary_text.tag_configure('title', font=('Helvetica', 11, 'bold'), foreground='#2c3e50')
        self.summary_text.tag_configure('value', font=('Monaco', 10, 'bold'), foreground=self.accent_color)
        self.summary_text.tag_configure('section', font=('Helvetica', 10, 'bold'), foreground='#e74c3c')

        # Matches card with notebook tabs
        match_frame = ttk.Labelframe(content_container, text="🔍 Match Results", padding=15, style='Card.TLabelframe')
        match_frame.grid(row=1, column=0, sticky="nsew")
        match_frame.columnconfigure(0, weight=1)
        match_frame.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(match_frame)
        notebook.grid(row=0, column=0, sticky="nsew", pady=(5, 0))

        fmf_tab = ttk.Frame(notebook)
        snpedia_tab = ttk.Frame(notebook)
        notebook.add(fmf_tab, text="Found My Fitness")
        notebook.add(snpedia_tab, text="SNPedia")

        # FMF matches
        fmf_tab.columnconfigure(0, weight=1)
        fmf_tab.rowconfigure(0, weight=1)
        fmf_columns = ("chrom", "rsid", "genotype", "fmf_genotype", "gene", "match_type", "note")
        self.fmf_table = ttk.Treeview(fmf_tab, columns=fmf_columns, show="headings", style='Modern.Treeview')
        self.fmf_table.grid(row=0, column=0, sticky="nsew")

        self.fmf_table.heading("chrom", text="Chr")
        self.fmf_table.heading("rsid", text="rsID")
        self.fmf_table.heading("genotype", text="Genotype")
        self.fmf_table.heading("fmf_genotype", text="FMF Genotype")
        self.fmf_table.heading("gene", text="Gene")
        self.fmf_table.heading("match_type", text="Match Type")
        self.fmf_table.heading("note", text="Note")

        self.fmf_table.column("chrom", width=60, anchor="center")
        self.fmf_table.column("rsid", width=110)
        self.fmf_table.column("genotype", width=90, anchor="center")
        self.fmf_table.column("fmf_genotype", width=110, anchor="center")
        self.fmf_table.column("gene", width=110)
        self.fmf_table.column("match_type", width=100, anchor="center")
        self.fmf_table.column("note", width=500)

        fmf_scroll = ttk.Scrollbar(fmf_tab, orient=tk.VERTICAL, command=self.fmf_table.yview)
        self.fmf_table.configure(yscrollcommand=fmf_scroll.set)
        fmf_scroll.grid(row=0, column=1, sticky="ns")

        self.fmf_table.tag_configure("match", background="#ffe6e6", foreground="#c0392b")
        self.fmf_table.tag_configure("complement", background="#e3f2fd", foreground="#1976d2")
        self.fmf_table.tag_configure("missing", background="#fff3e0", foreground="#e67e22")
        self.fmf_table.tag_configure("other", background="#f5f5f5", foreground="#555555")

        # SNPedia matches
        snpedia_tab.columnconfigure(0, weight=1)
        snpedia_tab.rowconfigure(1, weight=1)
        
        # SNPedia toolbar
        snpedia_toolbar = tk.Frame(snpedia_tab, bg=self.bg_color)
        snpedia_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        # High magnitude filter toggle
        ttk.Checkbutton(
            snpedia_toolbar,
            text="High magnitude only (≥ 1.5)",
            variable=self.snpedia_filter_var,
            command=self._on_toggle_high_mag,
            style='TCheckbutton'
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Threshold label and spinbox
        tk.Label(snpedia_toolbar, text="Threshold:", bg=self.bg_color, fg='#34495e').pack(side=tk.LEFT)
        self.mag_spinbox = tk.Spinbox(
            snpedia_toolbar,
            from_=0.0,
            to=5.0,
            increment=0.1,
            width=5,
            textvariable=self.snpedia_mag_threshold_var,
            command=self._on_mag_threshold_change,
        )
        self.mag_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        # Trigger handler on manual edits
        self.mag_spinbox.bind("<Return>", lambda e: self._on_mag_threshold_change())
        self.mag_spinbox.bind("<FocusOut>", lambda e: self._on_mag_threshold_change())
        if not self.snpedia_filter_high_mag:
            self.mag_spinbox.configure(state='disabled')

        # Sample size control for broader SNP fetches
        tk.Label(snpedia_toolbar, text="Sample size:", bg=self.bg_color, fg='#34495e').pack(side=tk.LEFT)
        self.sample_spinbox = tk.Spinbox(
            snpedia_toolbar,
            from_=200,
            to=5000,
            increment=100,
            width=6,
            textvariable=self.snpedia_sample_size_var,
        )
        self.sample_spinbox.pack(side=tk.LEFT, padx=(5, 10))

        # Top N display control
        tk.Label(snpedia_toolbar, text="Top N (0=All):", bg=self.bg_color, fg='#34495e').pack(side=tk.LEFT)
        self.topn_spinbox = tk.Spinbox(
            snpedia_toolbar,
            from_=0,
            to=5000,
            increment=50,
            width=6,
            textvariable=self.snpedia_top_n_var,
            command=self._on_top_n_change,
        )
        self.topn_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        self.topn_spinbox.bind("<Return>", lambda e: self._on_top_n_change())
        self.topn_spinbox.bind("<FocusOut>", lambda e: self._on_top_n_change())

        if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
            ttk.Button(snpedia_toolbar, text="🤖 Synthesize with AI", 
                      command=self._synthesize_snpedia_results,
                      style='Secondary.TButton').pack(side=tk.LEFT, padx=5)
        
        sp_columns = ("chrom", "rsid", "genotype", "magnitude", "repute", "summary")
        self.snpedia_table = ttk.Treeview(snpedia_tab, columns=sp_columns, show="headings", style='Modern.Treeview')
        self.snpedia_table.grid(row=1, column=0, sticky="nsew")

        self.snpedia_table.heading("chrom", text="Chr")
        self.snpedia_table.heading("rsid", text="rsID")
        self.snpedia_table.heading("genotype", text="Genotype")
        self.snpedia_table.heading("magnitude", text="Magnitude", command=self._toggle_snpedia_sort)
        self.snpedia_table.heading("repute", text="Repute")
        self.snpedia_table.heading("summary", text="Summary")

        self.snpedia_table.column("chrom", width=60, anchor="center")
        self.snpedia_table.column("rsid", width=110)
        self.snpedia_table.column("genotype", width=90, anchor="center")
        self.snpedia_table.column("magnitude", width=100, anchor="center")
        self.snpedia_table.column("repute", width=90, anchor="center")
        self.snpedia_table.column("summary", width=520)

        sp_scroll = ttk.Scrollbar(snpedia_tab, orient=tk.VERTICAL, command=self.snpedia_table.yview)
        self.snpedia_table.configure(yscrollcommand=sp_scroll.set)
        sp_scroll.grid(row=1, column=1, sticky="ns")

        # Double-click to open SNPedia page
        self.snpedia_table.bind("<Double-1>", self._on_snpedia_double_click)
        self.fmf_table.bind("<Double-1>", self._on_fmf_double_click)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select 23andMe Raw Data File",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*"),
            ],
        )
        if file_path:
            self.file_path.set(file_path)
            self.status.config(text="✅ Ready to analyze: {}".format(os.path.basename(file_path)))

    def fetch_snpedia_api(self):
        if not self.current_snps:
            messagebox.showinfo("Load genome first", "Please load and analyze a 23andMe file before fetching SNPedia.")
            return

        if self.fetch_thread and self.fetch_thread.is_alive():
            messagebox.showinfo("Fetch already running", "Please wait for the current SNPedia fetch to finish.")
            return

        # Offer to fetch only the most important SNPs (FMF list) to speed things up
        fmf_rsids = [k[1:] if k.startswith("*") else k for k in FMF_NOTEWORTHY.keys()]
        fmf_rsids = sorted({r for r in fmf_rsids if r in self.current_snps})

        # Determine broader sample size from UI (fallback to 500 if unset)
        try:
            sample_size = int(self.snpedia_sample_size_var.get())
        except Exception:
            sample_size = 500
        if sample_size < 200:
            sample_size = 200

        use_fmf_only = None
        if fmf_rsids:
            resp = messagebox.askyesnocancel(
                "Fetch SNPedia",
                "Prioritize FMF-noteworthy SNPs ({} rsIDs) and include more up to ~{} total?\n"
                "Choose Yes to fetch FMF + additional sampled SNPs (sorted by importance after fetch).\n"
                "Choose No to fetch a broader sample (~{}) without FMF prioritization.\n"
                "Cancel to stop.".format(len(fmf_rsids), sample_size, sample_size),
            )
            if resp is None:
                return
            use_fmf_only = resp
        else:
            # No FMF overlap; fall back to sample
            use_fmf_only = False

        if use_fmf_only:
            # Build list: FMF rsIDs plus additional sampled rsIDs up to the target sample size
            all_rsids = list(self.current_snps.keys())
            rs_snps = [r for r in all_rsids if r.startswith('rs')]
            # Exclude FMF from additional sampling
            remaining = [r for r in rs_snps if r not in fmf_rsids]
            target_total = max(sample_size, len(fmf_rsids))
            additional_needed = max(0, target_total - len(fmf_rsids))
            if additional_needed > 0 and remaining:
                step = max(1, len(remaining) // additional_needed)
                additional = [remaining[i] for i in range(0, len(remaining), step)][:additional_needed]
            else:
                additional = []
            # Combine FMF first, then additional sample
            rsids = fmf_rsids + additional
        else:
            # Fetch a broader sample of SNPs, prioritizing those likely to have SNPedia pages
            # Strategy: Take SNPs from common chromosomes and skip sequential dense regions
            all_rsids = list(self.current_snps.keys())

            # Filter to rs-numbered SNPs (more likely to be in SNPedia than i-numbered)
            rs_snps = [r for r in all_rsids if r.startswith('rs')]

            # Take every Nth SNP to sample across genome using configured sample_size (not just first 500)
            if len(rs_snps) > sample_size:
                step = max(1, len(rs_snps) // sample_size)
                rsids = [rs_snps[i] for i in range(0, len(rs_snps), step)][:sample_size]
            else:
                rsids = rs_snps

        if not rsids:
            messagebox.showinfo("No SNPs to fetch", "No SNPedia-eligible SNPs found to fetch.")
            return

        # Build (rsid, genotype) pairs for genotype-specific lookups
        snp_genotype_pairs = [(rsid, self.current_snps[rsid][2]) for rsid in rsids]

        # Init progress UI
        self.fetch_progress = 0
        self.fetch_total = len(snp_genotype_pairs)
        self.progress.config(mode="determinate", maximum=self.fetch_total, value=0)
        self.status.config(text="📡 Fetching {} SNPedia entries... this may take a bit.".format(len(snp_genotype_pairs)))
        self.update_idletasks()

        # Start background fetch
        self.fetch_thread = threading.Thread(target=self._snpedia_fetch_worker, args=(snp_genotype_pairs,), daemon=True)
        self.fetch_thread.start()
        self._poll_fetch_progress()

    def analyze_file(self):
        file_path = self.file_path.get().strip()
        if not file_path:
            messagebox.showwarning("No file selected", "Please select a 23andMe raw data file.")
            return

        try:
            snps = load_snps(file_path)
        except FileNotFoundError:
            messagebox.showerror("File not found", "Could not find the selected file.")
            return
        except ValueError as exc:
            messagebox.showerror("Invalid file", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error", "Failed to load file: {}".format(exc))
            return

        summary = compute_summary(snps)
        fmf_matches = find_fmf_matches(snps)

        snpedia_matches = []
        if self.snpedia_records:
            snpedia_matches = find_snpedia_matches(snps, self.snpedia_records)

        self.current_snps = snps
        self._render_summary(summary)
        self._render_fmf_matches(fmf_matches)
        self._render_snpedia_matches(snpedia_matches)

        status_bits = ["✨ Successfully loaded {:,} SNPs from {}".format(summary["total_snps"], os.path.basename(file_path))]
        status_bits.append("FMF matches: {}".format(len(fmf_matches)))
        if self.snpedia_records:
            status_bits.append("SNPedia matches: {} (sorted by magnitude)".format(len(snpedia_matches)))
        else:
            status_bits.append("Load a SNPedia TSV/CSV to see SNPedia matches")

        self.status.config(text=" | ".join(status_bits))

    def _render_summary(self, summary):
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)
        
        # Overview section
        self.summary_text.insert(tk.END, "📈 OVERVIEW\n", 'section')
        self.summary_text.insert(tk.END, "Total SNPs: ", 'title')
        self.summary_text.insert(tk.END, "{:,}\n".format(summary["total_snps"]), 'value')
        self.summary_text.insert(tk.END, "rsID count: ", 'title')
        self.summary_text.insert(tk.END, "{:,}  ".format(summary["rs_cnt"]), 'value')
        self.summary_text.insert(tk.END, "Internal ID count: ", 'title')
        self.summary_text.insert(tk.END, "{:,}\n".format(summary["internal_cnt"]), 'value')
        self.summary_text.insert(tk.END, "Unique positions: ", 'title')
        self.summary_text.insert(tk.END, "{:,}  ".format(summary["unique_positions"]), 'value')
        self.summary_text.insert(tk.END, "Unique genotypes: ", 'title')
        self.summary_text.insert(tk.END, "{}\n\n".format(summary["unique_genotypes"]), 'value')
        
        # Top genotypes
        self.summary_text.insert(tk.END, "🧬 TOP GENOTYPES\n", 'section')
        for genotype, count in summary["genotype_counts"][:8]:
            self.summary_text.insert(tk.END, "  {}: ".format(genotype), 'title')
            self.summary_text.insert(tk.END, "{:,}\n".format(count), 'value')
        
        # Chromosome distribution
        self.summary_text.insert(tk.END, "\n🧩 CHROMOSOME DISTRIBUTION\n", 'section')
        for chrom, count in summary["chromosome_counts"]:
            self.summary_text.insert(tk.END, "  Chr {}: ".format(chrom), 'title')
            self.summary_text.insert(tk.END, "{:,}\n".format(count), 'value')
        
        self.summary_text.config(state="disabled")

    def _clear_tree(self, table):
        for row in table.get_children():
            table.delete(row)

    def _render_fmf_matches(self, matches):
        self._clear_tree(self.fmf_table)

        for match in matches:
            self.fmf_table.insert(
                "",
                tk.END,
                values=(
                    match["chrom"],
                    match["rsid"],
                    match["genotype"],
                    match["fmf_genotype"],
                    match["gene"],
                    match["match_type"],
                    match["note"],
                ),
                tags=(match["match_type"],),
            )

    def _render_snpedia_matches(self, matches=None):
        if matches is not None:
            self.snpedia_matches = matches

        data = sorted(
            self.snpedia_matches,
            key=lambda m: m.get("magnitude", 0),
            reverse=self.snpedia_sort_desc,
        )

        # Apply high magnitude filter only when enabled
        if self.snpedia_filter_high_mag and self.snpedia_min_mag is not None:
            data = [m for m in data if (m.get("magnitude", 0) or 0) >= self.snpedia_min_mag]

        if self.snpedia_top_n is not None and len(data) > self.snpedia_top_n:
            data = data[: self.snpedia_top_n]

        self._clear_tree(self.snpedia_table)
        for match in data:
            self.snpedia_table.insert(
                "",
                tk.END,
                values=(
                    match.get("chrom", ""),
                    match.get("rsid", ""),
                    match.get("genotype", ""),
                    "{:.2f}".format(match.get("magnitude", 0)),
                    match.get("repute", ""),
                    match.get("summary", ""),
                ),
            )

    def _on_toggle_high_mag(self):
        self.snpedia_filter_high_mag = bool(self.snpedia_filter_var.get())
        # Enable/disable threshold spinbox
        if self.snpedia_filter_high_mag:
            self.mag_spinbox.configure(state='normal')
        else:
            self.mag_spinbox.configure(state='disabled')
        self._render_snpedia_matches()

    def _on_mag_threshold_change(self):
        try:
            val = float(self.snpedia_mag_threshold_var.get())
        except Exception:
            val = self.snpedia_min_mag or 1.5
        # Clamp value
        if val < 0.0:
            val = 0.0
        if val > 5.0:
            val = 5.0
        self.snpedia_mag_threshold_var.set(round(val, 1))
        self.snpedia_min_mag = val
        if self.snpedia_filter_high_mag:
            self._render_snpedia_matches()

    def _toggle_snpedia_sort(self):
        self.snpedia_sort_desc = not self.snpedia_sort_desc
        self._render_snpedia_matches()

    def _on_top_n_change(self):
        """Update the Top N cap for SNPedia table (0 means no cap)."""
        try:
            val = int(self.snpedia_top_n_var.get())
        except Exception:
            val = self.snpedia_top_n if isinstance(self.snpedia_top_n, int) else 500
        # Clamp and map 0 to None (no cap)
        if val <= 0:
            self.snpedia_top_n = None
            self.snpedia_top_n_var.set(0)
        else:
            if val > 5000:
                val = 5000
            self.snpedia_top_n = val
            self.snpedia_top_n_var.set(val)
        self._render_snpedia_matches()

    def _synthesize_snpedia_results(self):
        """Synthesize SNPedia results using OpenAI LLM."""
        if not self.snpedia_matches:
            messagebox.showinfo("No data", "Please fetch SNPedia data first.")
            return
        
        # Show a status message and disable button
        self.status.config(text="🤖 Synthesizing results with AI... please wait")
        self.update_idletasks()
        
        # Run synthesis in background thread
        thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        thread.start()
    
    def _synthesis_worker(self):
        """Background worker to call LLM and display results."""
        synthesis = synthesize_snpedia_with_llm(self.snpedia_matches, snp_dict=self.current_snps)
        self.after(0, self._show_synthesis_result, synthesis)
    
    def _show_synthesis_result(self, synthesis):
        """Display synthesis result in a new window with markdown and clickable links."""
        self.status.config(text="✨ Synthesis complete!")
        
        # Create a new window for the synthesis
        result_window = tk.Toplevel(self)
        result_window.title("AI-Synthesized Genetic Analysis")
        result_window.geometry("900x700")
        
        # Title
        title_frame = tk.Frame(result_window, bg=self.header_bg)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(title_frame, 
                              text="🧬 Your Personalized Genetic Analysis",
                              font=('Helvetica', 14, 'bold'),
                              bg=self.header_bg,
                              fg='white',
                              pady=15)
        title_label.pack()
        
        # Text widget with scrollbar
        text_frame = tk.Frame(result_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        text_widget = tk.Text(text_frame, 
                             font=('Helvetica', 11),
                             wrap=tk.WORD,
                             bg=self.card_bg,
                             fg='#333',
                             relief=tk.FLAT,
                             padx=15,
                             pady=15)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Configure text tags for markdown rendering
        text_widget.tag_configure("bold", font=('Helvetica', 11, 'bold'))
        text_widget.tag_configure("italic", font=('Helvetica', 11, 'italic'))
        text_widget.tag_configure("link", foreground='#0066cc', underline=True, font=('Helvetica', 11, 'underline'))
        text_widget.tag_configure("code", font=('Courier', 10), background='#f0f0f0')
        text_widget.tag_configure("heading", font=('Helvetica', 13, 'bold'), foreground=self.accent_color)
        
        # Parse and insert synthesis text with markdown support
        self._insert_markdown_text(text_widget, synthesis)
        text_widget.configure(state=tk.DISABLED)  # Read-only
        
        # Bind click events for hyperlinks
        text_widget.bind("<Button-1>", lambda e: self._on_text_click(e, text_widget, result_window))
        text_widget.bind("<Motion>", lambda e: self._on_text_motion(e, text_widget))
        
        # Close button
        btn_frame = tk.Frame(result_window, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, padx=15, pady=15)
        ttk.Button(btn_frame, text="Close", command=result_window.destroy,
                  style='Secondary.TButton').pack(side=tk.RIGHT)
    
    def _insert_markdown_text(self, text_widget, content):
        """Insert text with markdown parsing (bold, italic, links, headings)."""
        i = 0
        while i < len(content):
            # Check for markdown patterns
            if content[i:i+2] == "**" and i + 2 < len(content):
                # Bold text
                j = content.find("**", i + 2)
                if j != -1:
                    text_widget.insert(tk.END, content[i+2:j], "bold")
                    i = j + 2
                    continue
            
            if content[i:i+2] == "__" and i + 2 < len(content):
                # Italic text
                j = content.find("__", i + 2)
                if j != -1:
                    text_widget.insert(tk.END, content[i+2:j], "italic")
                    i = j + 2
                    continue
            
            if content[i] == "#" and (i == 0 or content[i-1] == "\n"):
                # Heading
                j = content.find("\n", i)
                if j == -1:
                    j = len(content)
                heading_text = content[i:j].lstrip("#").strip()
                text_widget.insert(tk.END, heading_text + "\n", "heading")
                i = j + 1
                continue
            
            if content[i] == "[" and "](" in content[i:]:
                # Link [text](url)
                close_bracket = content.find("]", i)
                open_paren = content.find("(", close_bracket)
                close_paren = content.find(")", open_paren)
                if close_bracket != -1 and open_paren != -1 and close_paren != -1:
                    link_text = content[i+1:close_bracket]
                    link_url = content[open_paren+1:close_paren]
                    text_widget.insert(tk.END, link_text, "link")
                    # Store URL in a tag
                    text_widget.tag_bind("link", "<Motion>", lambda e: text_widget.config(cursor="hand2"))
                    # We'll use a hack to store the URL - store it in a hidden marker
                    text_widget.insert(tk.END, f"\x00{link_url}\x00")
                    i = close_paren + 1
                    continue
            
            # Regular character
            text_widget.insert(tk.END, content[i])
            i += 1
    
    def _on_text_click(self, event, text_widget, window):
        """Handle clicks on hyperlinks in the text widget."""
        index = text_widget.index(f"@{event.x},{event.y}")
        
        # Check if we clicked on a link
        tags = text_widget.tag_names(index)
        if "link" in tags:
            # Get the line content and extract URL
            line_num = int(index.split(".")[0])
            line_start = f"{line_num}.0"
            line_end = f"{line_num}.end"
            line_content = text_widget.get(line_start, line_end)
            
            # Look for URL markers in nearby text
            full_text = text_widget.get("1.0", tk.END)
            # Find the URL associated with this link (search forward from current position)
            pos = full_text.find(line_content)
            if pos != -1:
                # Look for URL pattern after position
                match = re.search(r'\x00([^\x00]+)\x00', full_text[pos:])
                if match:
                    url = match.group(1)
                    if url.startswith("http"):
                        webbrowser.open(url)
    
    def _on_text_motion(self, event, text_widget):
        """Change cursor to hand when hovering over links."""
        index = text_widget.index(f"@{event.x},{event.y}")
        tags = text_widget.tag_names(index)
        if "link" in tags:
            text_widget.config(cursor="hand2")
        else:
            text_widget.config(cursor="arrow")

    def _snpedia_fetch_worker(self, snp_genotype_pairs):
        fetched = []
        errors = 0
        batch_size = 25  # 25 SNPs = 50 pages (rsID + genotype pages)
        
        # Process in batches
        for batch_start in range(0, len(snp_genotype_pairs), batch_size):
            batch = snp_genotype_pairs[batch_start:batch_start + batch_size]
            try:
                records = fetch_snpedia_batch(batch)
                if records:
                    fetched.extend(records)
            except Exception as e:
                errors += len(batch)
                # Continue processing other batches even if one fails
            
            # Update progress
            self.fetch_progress = min(batch_start + batch_size, len(snp_genotype_pairs))
            
        self.after(0, self._on_snpedia_fetch_complete, fetched, errors)

    def _poll_fetch_progress(self):
        if self.fetch_total:
            self.progress['value'] = min(self.fetch_progress, self.fetch_total)
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.after(100, self._poll_fetch_progress)

    def _on_snpedia_fetch_complete(self, fetched, errors):
        self.fetch_thread = None
        self.snpedia_records = fetched
        self.snpedia_sort_desc = True

        snpedia_matches = find_snpedia_matches(self.current_snps, self.snpedia_records)
        self._render_snpedia_matches(snpedia_matches)

        self.status.config(
            text="📡 SNPedia API fetched {} entries (errors: {}) | Matches: {}".format(
                len(fetched), errors, len(snpedia_matches)
            )
        )
        self.progress.config(value=0)

    def _open_snpedia_page(self, rsid):
        if not rsid:
            return
        url = f"https://www.snpedia.com/index.php/{rsid}"
        webbrowser.open(url)
        self.status.config(text=f"🌐 Opened SNPedia page for {rsid}")

    def _on_snpedia_double_click(self, event):
        row_id = self.snpedia_table.identify_row(event.y)
        if not row_id:
            return
        values = self.snpedia_table.item(row_id, "values")
        rsid = values[1] if len(values) > 1 else ""
        self._open_snpedia_page(rsid)

    def _on_fmf_double_click(self, event):
        row_id = self.fmf_table.identify_row(event.y)
        if not row_id:
            return
        values = self.fmf_table.item(row_id, "values")
        rsid = values[1] if len(values) > 1 else ""
        self._open_snpedia_page(rsid)


if __name__ == "__main__":
    app = SNPAnalyzerApp()
    app.mainloop()
