import math

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.particle_entity import ParticleEntity


class DEMEntity(ParticleEntity):
    """
    Entity for DEM (Discrete Element Method) particles.

    Particles are sampled on a face-centered cubic (FCC) lattice whose nearest-neighbor distance is
    `2.01 * radius`, matching the reference implementation's volume sampler. This guarantees zero initial
    overlap between grains, which the strictly elastic (undamped) contact model requires for stability.

    Parameters
    ----------
    scene : Scene
        The scene object that this entity belongs to.
    solver : DEMSolver
        The DEM solver responsible for simulating the entity's particles.
    material : DEM.Base
        The material definition, including density and contact properties.
    morph : Morph
        Geometry or volumetric shape used for sampling particles. Only primitive morphs are supported.
    surface : Surface
        Surface material or texture information associated with the entity.
    particle_size : float
        Size of each DEM particle (diameter).
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity's particles in the solver.
    name : str, optional
        Name of the entity.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        name=None,
    ):
        super().__init__(
            scene=scene,
            solver=solver,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=particle_size,
            idx=idx,
            particle_start=particle_start,
            need_skinning=False,
            name=name,
        )

    def sample(self):
        """
        Sample particles on an FCC lattice inside the primitive morph, following the reference volume sampler:
        the lattice cell size is `2.01 * radius * sqrt(2)`, and each cell contributes its lower corner plus its
        three lower face centers, giving a nearest-neighbor distance of `2.01 * radius`.
        """
        if not isinstance(self._morph, gs.options.morphs.Primitive):
            gs.raise_exception(f"DEMEntity only supports primitive morphs. Got: {self._morph}.")

        radius = self._particle_size / 2.0
        # C++ VolumeSampler: cell size = latticeConstant * sqrt(2), latticeConstant = 2.01 * radius
        cell_size = 2.01 * radius * math.sqrt(2.0)
        half_cell = 0.5 * cell_size

        lower, upper = self._vmesh.trimesh.bounds
        n_cells = np.maximum(np.ceil((upper - lower) / cell_size).astype(gs.np_int), 1) + 1
        cell_idx = np.indices(tuple(n_cells)).reshape(3, -1).T
        cell_origins = lower + cell_idx * cell_size
        fcc_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, half_cell, half_cell],
                [half_cell, 0.0, half_cell],
                [half_cell, half_cell, 0.0],
            ],
            dtype=gs.np_float,
        )
        candidates = (cell_origins[:, None, :] + fcc_offsets[None, :, :]).reshape(-1, 3)
        particles = candidates[self._vmesh.trimesh.contains(candidates)]

        if particles.shape[0] == 0:
            gs.raise_exception("Entity has zero particles.")

        # transform vmesh and particles by the morph pose (offset composed onto the morph pose)
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(self._morph.offset_pos, dtype=gs.np_float),
            np.array(self._morph.offset_quat, dtype=gs.np_float),
            np.array(self._morph.pos, dtype=gs.np_float),
            np.array(self._morph.quat, dtype=gs.np_float),
        )
        self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        particles = gu.transform_by_trans_quat(particles, pos, quat)

        if not self._solver.boundary.is_inside(particles):
            gs.raise_exception(
                "Entity has particles outside solver boundary.\n\nCurrent boundary:\n"
                f"{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
            )

        self._vverts = np.zeros((0, 3), dtype=gs.np_float)
        self._vfaces = np.zeros((0, 3), dtype=gs.np_int)

        self._particles = np.asarray(particles, dtype=gs.np_float, order="C")
        self._init_particles_offset = gs.tensor(self._particles) - gs.tensor(pos)
        self._n_particles = len(self._particles)

        gs.logger.info(f"Sampled ~~<{self._n_particles:,}>~~ DEM particles (FCC lattice).")

    def _add_particles_to_solver(self):
        self.solver._kernel_add_particles(
            particle_start=self._particle_start,
            n_particles=self._n_particles,
            positions=self._particles,
            radius=self._particle_size / 2.0,
            density=self._material.rho,
        )

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_pos(particles_idx_local_, envs_idx, poss)

    @gs.assert_built
    def _set_particles_pos_grad(self, poss_grad):
        pass

    @gs.assert_built
    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_pos(self._particle_start, self._n_particles, envs_idx, poss)
        return poss

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_vel(particles_idx_local_, envs_idx, vels)

    @gs.assert_built
    def _set_particles_vel_grad(self, vels_grad):
        pass

    @gs.assert_built
    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_vel(self._particle_start, self._n_particles, envs_idx, vels)
        return vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_active(particles_idx_local_, envs_idx, actives)

    @gs.assert_built
    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = torch.zeros((len(envs_idx), self._n_particles), dtype=gs.tc_bool, device=gs.device)
        self.solver._kernel_get_particles_active(self._particle_start, self._n_particles, envs_idx, actives)
        return actives
