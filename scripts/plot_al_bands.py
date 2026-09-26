#!/usr/bin/env python3
"""Plot the A5 fcc-Al bands relative to the Fermi level from the A5 SCF run."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


PROJECT = Path(__file__).resolve().parents[1]
BOHR_TO_ANGSTROM = 0.529177210903
A0_ANGSTROM = 4.02496285
ALAT_BOHR = A0_ANGSTROM / BOHR_TO_ANGSTROM
HARTREE_TO_EV = 27.211386245988
VALENCE_ELECTRONS = 3

# Persistent report palette: navy = PBE, gold = comparison/empty-lattice, gray = reference.
PBE_NAVY = "#253E75"
EMPTY_LATTICE_GOLD = "#D3B54A"
REFERENCE_GRAY = "#5A5A5A"

FERMI_PATTERN = re.compile(
    r"the\s+Fermi\s+energy\s+is\s+([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+ev",
    re.IGNORECASE,
)
NBND_PATTERN = re.compile(r"number\s+of\s+Kohn-Sham\s+states\s*=\s*(\d+)", re.IGNORECASE)
KPOINT_PATTERN = re.compile(
    r"^\s*k\s*=\s*"
    r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s*"
    r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)"
    r".*bands\s*\(ev\):",
    re.IGNORECASE,
)


def qe_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def require_finished(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")
    text = path.read_text(errors="replace")
    if "JOB DONE." not in text:
        raise RuntimeError(f"QE did not finish cleanly: {path}")
    return text


def read_fermi_energy(scf_output: Path) -> float:
    matches = FERMI_PATTERN.findall(require_finished(scf_output))
    if not matches:
        raise ValueError(f"Could not find 'the Fermi energy is ... ev' in {scf_output}")
    return qe_float(matches[-1])


def read_band_eigenvalues(bands_output: Path) -> tuple[np.ndarray, np.ndarray]:
    text = require_finished(bands_output)
    state_matches = NBND_PATTERN.findall(text)
    if not state_matches:
        raise ValueError(f"Could not determine nbnd from {bands_output}")
    nbnd = int(state_matches[-1])

    kpoints: list[tuple[float, float, float]] = []
    rows: list[list[float]] = []
    current_k: tuple[float, float, float] | None = None
    current_energies: list[float] = []

    for line in text.splitlines():
        match = KPOINT_PATTERN.match(line)
        if match:
            if current_k is not None:
                raise ValueError("Encountered a new k-point before collecting all its eigenvalues.")
            current_k = tuple(qe_float(value) for value in match.groups())
            current_energies = []
            continue

        if current_k is None:
            continue

        tokens = line.split()
        if not tokens:
            continue
        try:
            current_energies.extend(qe_float(token) for token in tokens)
        except ValueError:
            continue

        if len(current_energies) == nbnd:
            kpoints.append(current_k)
            rows.append(current_energies)
            current_k = None
        elif len(current_energies) > nbnd:
            raise ValueError("Parsed more eigenvalues than the reported number of bands.")

    if current_k is not None:
        raise ValueError("The last k-point did not contain a complete set of eigenvalues.")
    if not rows:
        raise ValueError(f"No band eigenvalues were parsed from {bands_output}")
    return np.asarray(kpoints, dtype=float), np.asarray(rows, dtype=float)


def read_kpath(kpath_csv: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if not kpath_csv.is_file():
        raise FileNotFoundError(
            f"Path table not found: {kpath_csv}. Run 'python3 scripts/al_bands.py' first."
        )
    rows = list(csv.DictReader(kpath_csv.open()))
    if not rows:
        raise ValueError(f"No k-path rows found in {kpath_csv}")

    distance_2pi_over_a = np.asarray([float(row["distance_2pi_over_a"]) for row in rows])
    cartesian_kpoints = np.asarray(
        [
            [
                float(row["kx_cart_2pi_over_a"]),
                float(row["ky_cart_2pi_over_a"]),
                float(row["kz_cart_2pi_over_a"]),
            ]
            for row in rows
        ]
    )
    labels = [row["label"] for row in rows]
    special_indices = np.asarray([index for index, label in enumerate(labels) if label])
    return distance_2pi_over_a, cartesian_kpoints, labels, special_indices


def free_electron_bands(
    cartesian_kpoints_2pi_over_a: np.ndarray,
    number_of_bands: int,
) -> tuple[np.ndarray, float, float, int]:
    """Return the relevant fcc empty-lattice parabolas relative to ``E_F^free``.

    The direct fcc primitive vectors imply a bcc reciprocal lattice.  We enumerate
    reciprocal vectors as integer combinations of the reciprocal primitive vectors
    and retain every parabola that contributes to one of the lowest
    ``number_of_bands`` folded free-electron bands at any path point.  This is a
    principled finite set for comparison with the ``nbnd`` Kohn--Sham bands,
    rather than an arbitrary visual truncation of reciprocal vectors.
    """
    volume_bohr3_per_atom = ALAT_BOHR**3 / 4.0
    electron_density_bohr3 = VALENCE_ELECTRONS / volume_bohr3_per_atom
    fermi_free_ev = (
        0.5 * (3.0 * np.pi**2 * electron_density_bohr3) ** (2.0 / 3.0) * HARTREE_TO_EV
    )

    # Rows are b_1, b_2, b_3 in Cartesian coordinates.  Their units are bohr^-1.
    reciprocal_primitive = (2.0 * np.pi / ALAT_BOHR) * np.asarray(
        [
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ]
    )
    kpoints_bohr_inv = cartesian_kpoints_2pi_over_a * (2.0 * np.pi / ALAT_BOHR)

    # Expand the candidate set if a selected reciprocal vector lies on its edge.
    for integer_limit in range(2, 7):
        coefficients = np.asarray(
            [
                (h, k, l)
                for h in range(-integer_limit, integer_limit + 1)
                for k in range(-integer_limit, integer_limit + 1)
                for l in range(-integer_limit, integer_limit + 1)
            ],
            dtype=int,
        )
        reciprocal_vectors = coefficients @ reciprocal_primitive
        energy_ev = (
            0.5
            * np.sum(
                (kpoints_bohr_inv[np.newaxis, :, :] + reciprocal_vectors[:, np.newaxis, :]) ** 2,
                axis=2,
            )
            * HARTREE_TO_EV
        )
        # Include every exact degeneracy at the ``number_of_bands`` boundary;
        # otherwise np.argpartition would select an arbitrary subset of equal
        # free-electron parabolas at a high-symmetry point.
        energy_at_boundary = np.partition(energy_ev, number_of_bands - 1, axis=0)[
            number_of_bands - 1
        ]
        selected = np.flatnonzero(
            np.any(energy_ev <= energy_at_boundary[np.newaxis, :] + 1.0e-10, axis=1)
        )
        if not np.any(np.abs(coefficients[selected]) == integer_limit):
            selected = selected[np.argsort(np.linalg.norm(reciprocal_vectors[selected], axis=1))]
            return (
                energy_ev[selected] - fermi_free_ev,
                fermi_free_ev,
                electron_density_bohr3,
                len(selected),
            )

    raise RuntimeError("Could not converge the reciprocal-vector set for the empty-lattice bands.")


def make_figure(
    distance_bohr_inv: np.ndarray,
    labels: list[str],
    special_indices: np.ndarray,
    energies_ev: np.ndarray,
    fermi_ev: float,
    empty_lattice_relative_ev: np.ndarray,
    output: Path,
) -> None:
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

    fig, axis = plt.subplots(figsize=(7.6, 4.6), dpi=900)
    dft_relative_ev = energies_ev - fermi_ev
    y_padding_ev = 1.0
    axis.set_ylim(dft_relative_ev.min() - y_padding_ev, dft_relative_ev.max() + y_padding_ev)

    # Draw empty-lattice curves first, so the Kohn--Sham bands remain legible.
    for band in empty_lattice_relative_ev:
        axis.plot(
            distance_bohr_inv,
            band,
            color=EMPTY_LATTICE_GOLD,
            linestyle=(0, (5, 3)),
            linewidth=1.05,
            alpha=0.88,
            zorder=1,
        )
    for band in energies_ev.T:
        axis.plot(distance_bohr_inv, band - fermi_ev, color=PBE_NAVY, linewidth=1.20, zorder=3)

    axis.plot([], [], color=PBE_NAVY, linewidth=1.55, label="PBE Kohn–Sham bands")
    axis.plot(
        [],
        [],
        color=EMPTY_LATTICE_GOLD,
        linestyle=(0, (5, 3)),
        linewidth=1.55,
        label=r"Empty-lattice bands, $E-E_F^{\mathrm{free}}$",
    )
    axis.axhline(
        0.0,
        color=REFERENCE_GRAY,
        linestyle="--",
        linewidth=1.10,
        label=r"Fermi-reference zero, $E-E_F=0$",
    )

    high_symmetry_distance = distance_bohr_inv[special_indices]
    for location in high_symmetry_distance:
        axis.axvline(location, color=REFERENCE_GRAY, linewidth=0.70, alpha=0.55, zorder=0)

    axis.set_xticks(high_symmetry_distance, [labels[index] for index in special_indices])
    axis.set_xlabel(r"Cumulative reciprocal-space distance, $s$ (bohr$^{-1}$)")
    axis.set_ylabel(r"Band energy, $E-E_F$ (eV)")
    axis.xaxis.set_minor_locator(AutoMinorLocator(4))
    axis.yaxis.set_minor_locator(AutoMinorLocator(4))
    axis.tick_params(which="major", direction="in", top=True, right=True, length=7, width=1.25)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=3.5, width=0.90)
    axis.margins(x=0.015)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.35,
    )
    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=900, bbox_inches="tight", pad_inches=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scf-output", type=Path, default=PROJECT / "qe-results" / "Al.band.scf.out")
    parser.add_argument("--bands-output", type=Path, default=PROJECT / "qe-results" / "Al.band.out")
    parser.add_argument("--kpath", type=Path, default=PROJECT / "a5-bands-kpath.csv")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "A5_band_structure_free_electron.png"
    )
    arguments = parser.parse_args()

    fermi_ev = read_fermi_energy(arguments.scf_output)
    qe_kpoints, energies_ev = read_band_eigenvalues(arguments.bands_output)
    distance_2pi_over_a, expected_kpoints, labels, special_indices = read_kpath(arguments.kpath)

    if len(qe_kpoints) != len(expected_kpoints):
        raise ValueError(
            f"QE returned {len(qe_kpoints)} k-points but the generated path has "
            f"{len(expected_kpoints)}. Confirm that Al.band.inp was regenerated before running."
        )
    coordinate_error = np.linalg.norm(qe_kpoints - expected_kpoints, axis=1).max()
    if coordinate_error > 2.0e-4:
        raise ValueError(
            f"QE k-point order does not match the generated fcc path "
            f"(maximum coordinate error {coordinate_error:.3e} in 2π/a)."
        )

    distance_bohr_inv = distance_2pi_over_a * 2.0 * np.pi / ALAT_BOHR
    empty_lattice_relative_ev, fermi_free_ev, density_bohr3, number_of_parabolas = (
        free_electron_bands(expected_kpoints, energies_ev.shape[1])
    )
    make_figure(
        distance_bohr_inv,
        labels,
        special_indices,
        energies_ev,
        fermi_ev,
        empty_lattice_relative_ev,
        arguments.output,
    )
    print(f"Fermi energy from SCF: {fermi_ev:.6f} eV")
    print(f"PBE equilibrium primitive-cell volume: {ALAT_BOHR**3 / 4.0:.6f} bohr^3/atom")
    print(f"Valence-electron density n = 3/V: {density_bohr3:.8f} bohr^-3")
    print(f"Free-electron Fermi energy: {fermi_free_ev:.6f} eV")
    print(
        f"Overlay uses {number_of_parabolas} reciprocal-vector parabolas that contribute "
        f"to the lowest {energies_ev.shape[1]} folded free-electron bands."
    )
    print(f"Parsed {len(qe_kpoints)} k-points and {energies_ev.shape[1]} bands.")
    print(f"Maximum k-point mismatch: {coordinate_error:.3e} in 2π/a")
    print(f"Wrote {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
