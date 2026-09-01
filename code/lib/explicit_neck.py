"""Three-dimensional ohmic neck model and static conductance extraction."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import meshio
import numpy as np


def write_cylinder_geo(path: Path, length: float, radius: float, mesh_size: float) -> None:
    path.write_text(
        f"""SetFactory(\"OpenCASCADE\");
Cylinder(1) = {{0, 0, 0, {length:.16g}, 0, 0, {radius:.16g}, 2*Pi}};
MeshSize {{ PointsOf{{ Volume{{1}}; }} }} = {mesh_size:.16g};
Physical Volume(\"neck\", 10) = {{1}};
"""
    )


def mesh_cylinder(geo: Path, msh: Path) -> None:
    gmsh = shutil.which("gmsh") or "/opt/anaconda3/bin/gmsh"
    command = [gmsh, str(geo), "-3", "-format", "msh2", "-o", str(msh)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"}
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode or not msh.exists():
        raise RuntimeError(f"Gmsh failed: {result.stderr[-2000:]}")


def load_tet_mesh(path: Path):
    from skfem.io.meshio import from_meshio

    raw = meshio.read(path)
    block = next(cell for cell in raw.cells if cell.type.startswith("tetra"))
    return from_meshio(meshio.Mesh(points=raw.points[:, :3], cells=[("tetra", block.data)]))


def solve_neck_conductance(mesh, conductivity: float, length: float) -> dict:
    """Solve a unit-voltage 3-D neck; side walls are naturally insulating."""
    from skfem import Basis, BilinearForm, ElementTetP1, Functional, asm, condense, solve
    from skfem.helpers import dot, grad

    basis = Basis(mesh, ElementTetP1())

    @BilinearForm
    def conduction(u, v, w):
        return conductivity * dot(grad(u), grad(v))

    stiffness = asm(conduction, basis)
    tolerance = max(1e-12, length * 1e-8)
    left = basis.get_dofs(lambda x: x[0] < tolerance).all()
    right = basis.get_dofs(lambda x: x[0] > length - tolerance).all()
    prescribed = np.unique(np.r_[left, right])
    values = np.zeros(basis.N)
    values[right] = 1.0
    potential = solve(*condense(stiffness, np.zeros(basis.N), x=values, D=prescribed))

    @Functional
    def energy(w):
        return conductivity * dot(w["u"].grad, w["u"].grad)

    conductance_energy = float(asm(energy, basis, u=potential))
    residual = stiffness @ potential
    conductance_left = float(-residual[left].sum())
    conductance_right = float(residual[right].sum())
    return {
        "conductance_energy_S": conductance_energy,
        "conductance_left_S": conductance_left,
        "conductance_right_S": conductance_right,
        "terminal_mismatch": abs(conductance_left - conductance_right)
        / max(conductance_energy, 1e-30),
        "nodes": int(mesh.nvertices),
        "elements": int(mesh.nelements),
    }


def analytic_cylinder_conductance(conductivity: float, length: float, radius: float) -> float:
    return conductivity * math.pi * radius**2 / length

