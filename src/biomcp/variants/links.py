"""Variant link generation functionality."""

from typing import Optional


def generate_dbsnp_url(rsid: str) -> Optional[str]:
    """Generate dbSNP URL for a variant."""
    if not rsid:
        return None
    return f"https://www.ncbi.nlm.nih.gov/snp/{rsid}"


def generate_clinvar_url(variant_id: str) -> Optional[str]:
    """Generate ClinVar URL for a variant."""
    if not variant_id:
        return None
    return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant_id}"


def generate_ensembl_url(variant_id: str) -> Optional[str]:
    """Generate Ensembl URL for a variant."""
    if not variant_id:
        return None
    return f"https://ensembl.org/Homo_sapiens/Variation/Explore?v={variant_id}"


def generate_cosmic_url(variant_id: str) -> Optional[str]:
    """Generate COSMIC URL for a variant."""
    if not variant_id:
        return None
    return f"https://cancer.sanger.ac.uk/cosmic/mutation/overview?id={variant_id}"


def generate_civic_url(variant_id: str) -> Optional[str]:
    """Generate CIViC URL for a variant."""
    if not variant_id:
        return None
    return f"https://civicdb.org/variants/{variant_id}"


def generate_genenames_url(gene: str) -> Optional[str]:
    """Generate GeneNames URL for a gene."""
    if not gene:
        return None
    return f"https://www.genenames.org/data/gene-symbol-report/#!/symbol/{gene}"


def generate_ucsc_url(chromosome: str, position: int) -> Optional[str]:
    """Generate UCSC Genome Browser URL for a variant."""
    if not chromosome or not position:
        return None
    return f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&position=chr{chromosome}:{position}"


def generate_all_links(variant_data: dict) -> dict:
    """Generate all available links for a variant."""
    links = {}
    
    if rsid := variant_data.get("rsid"):
        links["dbsnp"] = generate_dbsnp_url(rsid)
    
    if variant_id := variant_data.get("variant_id"):
        links["clinvar"] = generate_clinvar_url(variant_id)
        links["ensembl"] = generate_ensembl_url(variant_id)
        links["cosmic"] = generate_cosmic_url(variant_id)
        links["civic"] = generate_civic_url(variant_id)
    
    if gene := variant_data.get("gene"):
        links["genenames"] = generate_genenames_url(gene)
    
    if chromosome := variant_data.get("chromosome"):
        if position := variant_data.get("position"):
            links["ucsc"] = generate_ucsc_url(chromosome, position)
    
    return {k: v for k, v in links.items() if v is not None}