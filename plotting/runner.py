from __future__ import annotations

from plotting.config import CFG, PlotConfig
from plotting.figures.appendix import *
from plotting.figures.core import *
from plotting.figures.spectra import *
from plotting.figures.summary import *
from plotting.io.loaders import load_npz_dict, load_train_acc_npz, resolve_eos_path
from plotting.io.protocols import align_eos_to_entries, align_train_acc_to_entries, extract_eos_series, get_protocol_entries, get_variant_names
from plotting.style.base import setup_style


def run_all_figures(cfg: PlotConfig = CFG) -> None:
    cfg.prepare_dirs()
    setup_style()

    mf = load_npz_dict(cfg.mf_npz_path)
    entries = get_protocol_entries(mf, cfg.primary_protocol)
    if len(entries) == 0:
        raise RuntimeError(f"No entries found for protocol {cfg.primary_protocol}")

    train_payload = load_train_acc_npz(cfg.train_acc_npz_path)
    train_aligned = align_train_acc_to_entries(entries, train_payload)

    variants = get_variant_names(entries, cfg.primary_family)
    if len(variants) == 0:
        raise RuntimeError(f"No variants found for family {cfg.primary_family}")
    variant_name = cfg.primary_variant if cfg.primary_variant in variants else variants[0]

    eos_path = resolve_eos_path()
    eos_payload = load_npz_dict(eos_path) if eos_path is not None else None
    eos_series = extract_eos_series(eos_payload)
    eos_aligned = align_eos_to_entries(entries, eos_series) if len(eos_series) else {}

    fig01a_core_claim_mf_eos_test(entries, variant_name, eos_aligned, train_aligned)
    fig01b_core_claim_train_test(entries, variant_name, train_aligned)
    fig02_temporal_dynamics(entries, variant_name, eos_aligned, train_aligned)
    fig03_tau_structure(entries, variant_name)
    fig04_alpha_distributions(entries, variant_name)
    fig05_signed_q_asymmetry(entries, variant_name)
    fig06_triptych(entries, variant_name)
    fig07_phase_diagram(entries, variant_name, eos_aligned)
    fig08_train_test_accuracy(entries, train_aligned)

    figA1_variant_comparison(entries, variants)
    figA2_batch_robustness(entries, variant_name)
    figA3_bank_spaghetti(entries, variant_name)
    figA4_selected_times_tau_f_alpha(entries, variant_name)
    figA5_eos_vs_mf_scatter(entries, variant_name, eos_aligned)
    figA6_metadata(entries, eos_path, variant_name, train_aligned)

    print(f"All figures saved to: {cfg.out_dir}")
