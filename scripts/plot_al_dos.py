#!/usr/bin/env python3
"""Plot the A5 total DOS from QE dos.x relative to the NSCF Fermi level."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


PROJECT = Path(__file__).resolve().parents[1]
PBE_NAVY = "#253E75"
REFERENCE_GRAY = "#5A5A5A"
FERMI_PATTERN = re.compile(
    r"the\s+Fermi\s+energy\s+is\s+([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+ev",
    re.IGNORECASE,
)


def require_finished(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")
    text = path.read_text(errors="replace")
    if "JOB DONE." not in text:
        raise RuntimeError(f"QE did not finish cleanly: {path}")
    return text


def read_fermi_energy(nscf_output: Path) -> float:
    matches = FERMI_PATTERN.findall(require_finished(nscf_output))
    if not matches:
        raise ValueError(f"Could not find the NSCF Fermi energy in {nscf_output}")
    return float(matches[-1].replace("D", "E").replace("d", "e"))


def read_dos(dos_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dos_file.is_file():
        raise FileNotFoundError(f"DOS table not found: {dos_file}")
    try:
        data = np.loadtxt(dos_file, comments="#")
    except ValueError as error:
        raise ValueError(f"Could not parse numeric DOS columns in {dos_file}") from error
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected energy and DOS columns in {dos_file}")
    if np.any(data[:, 1] < -1.0e-10):
        raise ValueError("The total DOS contains unphysical negative values.")
    return data[:, 0], data[:, 1]


def make_figure(energy_ev: np.ndarray, dos: np.ndarray, fermi_ev: float, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 1.3,
            "mathtext.fontset": "dejavusans",
        }
    )

    energy_relative_ev = energy_ev - fermi_ev
    fig, axis = plt.subplots(figsize=(7.6, 4.6), dpi=900)
    axis.plot(energy_relative_ev, dos, color=PBE_NAVY, linewidth=1.65, label="PBE total DOS")
    axis.axvline(
        0.0,
        color=REFERENCE_GRAY,
        linestyle="--",
        linewidth=1.10,
        label=r"Fermi level, $E-E_F=0$",
    )
    axis.set_xlabel(r"Energy relative to Fermi level, $E-E_F$ (eV)")
    axis.set_ylabel(r"Density of states, $D(E)$ (states eV$^{-1}$ atom$^{-1}$)")
    axis.set_xlim(energy_relative_ev.min(), energy_relative_ev.max())
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_minor_locator(AutoMinorLocator(4))
    axis.yaxis.set_minor_locator(AutoMinorLocator(4))
    axis.tick_params(which="major", direction="in", top=True, right=True, length=7, width=1.25)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=3.5, width=0.90)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.7,
    )
    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=900, bbox_inches="tight", pad_inches=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nscf-output", type=Path, default=PROJECT / "qe-results" / "Al.dos.nscf.out"
    )
    parser.add_argument("--dos-output", type=Path, default=PROJECT / "qe-results" / "Al.dos.out")
    parser.add_argument("--dos-file", type=Path, default=PROJECT / "qe-results" / "Al.dos.dat")
    parser.add_argument("--output", type=Path, default=PROJECT / "A5_density_of_states.png")
    arguments = parser.parse_args()

    require_finished(arguments.dos_output)
    fermi_ev = read_fermi_energy(arguments.nscf_output)
    energy_ev, dos = read_dos(arguments.dos_file)
    make_figure(energy_ev, dos, fermi_ev, arguments.output)
    print(f"NSCF Fermi energy: {fermi_ev:.6f} eV")
    print(f"Read {len(energy_ev)} DOS points from {arguments.dos_file.resolve()}")
    print(f"Wrote {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
