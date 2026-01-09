#!/usr/bin/env python3

"""Simple GUI for analyzing 23andMe SNP data."""

import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from foundMyFitness import FMF_NOTEWORTHY


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
        ttk.Button(file_frame, text="📥 Load SNPedia", command=self.load_snpedia_file, style='Secondary.TButton').grid(row=0, column=4, padx=5)

        # Status bar with subtle background
        status_frame = tk.Frame(self, bg='#ecf0f1', pady=8)
        status_frame.grid(row=1, column=0, sticky="ew")
        self.status = tk.Label(status_frame,
                              text="👋 Welcome! Select a 23andMe raw data file to begin analysis.",
                              font=('Helvetica', 10),
                              bg='#ecf0f1',
                              fg='#34495e')
        self.status.pack(padx=20, anchor='w')

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
        snpedia_tab.rowconfigure(0, weight=1)
        sp_columns = ("chrom", "rsid", "genotype", "magnitude", "repute", "summary")
        self.snpedia_table = ttk.Treeview(snpedia_tab, columns=sp_columns, show="headings", style='Modern.Treeview')
        self.snpedia_table.grid(row=0, column=0, sticky="nsew")

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
        sp_scroll.grid(row=0, column=1, sticky="ns")

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

    def load_snpedia_file(self):
        file_path = filedialog.askopenfilename(
            title="Select SNPedia TSV or CSV export",
            filetypes=[
                ("TSV files", "*.tsv"),
                ("CSV files", "*.csv"),
                ("All files", "*"),
            ],
        )
        if not file_path:
            return

        try:
            records = load_snpedia_records(file_path)
        except Exception as exc:
            messagebox.showerror("Error", "Failed to load SNPedia file: {}".format(exc))
            return

        if not records:
            messagebox.showwarning("No records", "The selected SNPedia file contained no records.")
            return

        self.snpedia_records = records
        self.snpedia_file.set(os.path.basename(file_path))
        self.snpedia_sort_desc = True

        snpedia_matches = []
        if self.current_snps:
            snpedia_matches = find_snpedia_matches(self.current_snps, self.snpedia_records)
        self._render_snpedia_matches(snpedia_matches)

        self.status.config(
            text="📥 Loaded {} SNPedia records from {}{}".format(
                len(self.snpedia_records),
                os.path.basename(file_path),
                " | Re-run analysis to refresh matches" if not self.current_snps else "",
            )
        )

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

    def _toggle_snpedia_sort(self):
        self.snpedia_sort_desc = not self.snpedia_sort_desc
        self._render_snpedia_matches()


if __name__ == "__main__":
    app = SNPAnalyzerApp()
    app.mainloop()
