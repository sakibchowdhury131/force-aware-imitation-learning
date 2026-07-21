"""
Contact detection from Kinova Jaco2 joint torques.

Reusable logic (no hardware dependency) shared by diag_contact.py (live demo)
and, later, anything that wants to react to contact in real time (e.g. a
safety stop in 07_deploy.py, or auto-labeling contact frames during
replay_episode.py). Kept as an importable module — unlike the fully
self-contained diagnostic scripts — because this logic is meant to be reused,
the same way policy_common.py is shared by 05_train.py and test_policy.py.

Detection is based on the gravity-free ("external load") joint torque, mapped
to an end-effector force via the Jacobian transpose (see diag_joint_torques.py
for the derivation and a finite-difference check of compute_jacobian):

    tau_gf = J^T @ F   =>   F = pinv(J^T) @ tau_gf

Contact is flagged when ||F_xyz|| crosses a threshold, with:
  - baseline calibration (subtracts the residual torque bias measured while
    stationary and NOT in contact — the gravity model is never perfect)
  - hysteresis (exit threshold < enter threshold, avoids chattering right at
    the boundary)
  - debounce (requires N consecutive samples before flipping state, filters
    single-sample sensor spikes)
"""
import numpy as np
from scipy.spatial.transform import Rotation

# ════════════════════════════════════════════════════════════════════════════
# Forward kinematics + geometric Jacobian — same DH chain as 07_deploy.py /
# deploy_viz.py / diag_joint_torques.py (verified there against finite
# differences to ~3e-7).
# ════════════════════════════════════════════════════════════════════════════

_PI = np.pi
_FK_JOINT_PARAMS = [
    ([0,       0,       0.15675], [0,      _PI,   0    ]),
    ([0,       0.0016, -0.11875], [-_PI/2, 0,     _PI  ]),
    ([0,      -0.410,  0       ], [0,      _PI,   0    ]),
    ([0,       0.2073, -0.0114 ], [-_PI/2, 0,     _PI  ]),
    ([0,       0,      -0.10375], [ _PI/2, 0,     _PI  ]),
    ([0,       0.10375, 0      ], [-_PI/2, 0,     _PI  ]),
]
_FK_EEF_PARAMS = ([0, 0, -0.1600], [_PI, 0, _PI/2])


def _make_fk_T(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def fk_frames(q_deg: np.ndarray):
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    pre_frames = []
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        T_pre = T @ _make_fk_T(xyz, rpy)
        pre_frames.append(T_pre)
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T_pre @ Tj
    T_eef = T @ _make_fk_T(*_FK_EEF_PARAMS)
    return pre_frames, T_eef


def compute_jacobian(q_deg: np.ndarray) -> np.ndarray:
    pre_frames, T_eef = fk_frames(q_deg)
    p_eef = T_eef[:3, 3]
    J = np.zeros((6, 6))
    for i, T_pre in enumerate(pre_frames):
        z_i = T_pre[:3, 2]
        p_i = T_pre[:3, 3]
        J[:3, i] = np.cross(z_i, p_eef - p_i)
        J[3:, i] = z_i
    return J


def torque_to_wrench(q_deg: np.ndarray, tau: np.ndarray,
                     damping: float = 0.0, warn_cond_threshold: float = 1e4) -> np.ndarray:
    """Returns [Fx,Fy,Fz,Mx,My,Mz] at the EEF origin, base frame.

    Near a kinematic singularity, cond(J) blows up and plain pinv(J^T)
    amplifies torque noise without bound (confirmed on hardware — a J5=180deg
    spherical-wrist singularity turned a real ~1N·m torque error into a
    reported multi-million-Newton "force"). Two independent safeguards:
      - damping > 0 uses damped least squares instead of pinv: this is the
        Tikhonov-regularized solution to min ||J^T F - tau||^2 + damping^2||F||^2,
        i.e. F = (J J^T + damping^2 I)^-1 J tau. damping=0 (default) is
        unchanged plain-pinv behavior, so existing callers are unaffected.
      - Always prints a one-line warning if cond(J) exceeds warn_cond_threshold,
        regardless of damping, since a huge condition number means the
        reported force should not be trusted even if damping softened it.
      - mode='joint' on ContactDetector remains the recommended fallback near
        singularities (bypasses this Jacobian mapping entirely).
    """
    J = compute_jacobian(q_deg)
    cond = np.linalg.cond(J)
    if cond > warn_cond_threshold:
        print(f'  WARNING [torque_to_wrench]: cond(J)={cond:.2e} at this pose — '
             f'near a kinematic singularity, force estimate is unreliable. '
             f'Consider mode="joint" instead.')
    if damping <= 0.0:
        return np.linalg.pinv(J.T) @ tau
    return np.linalg.solve(J @ J.T + damping**2 * np.eye(6), J @ tau)


# ════════════════════════════════════════════════════════════════════════════
# Rigid-body dynamics (Recursive Newton-Euler, gravity excluded)
#
# tau_measured (gravity-free) = M(q)*qddot + C(q,qdot)*qdot + tau_ext
#
# GetAngularForceGravityFree already strips G(q). This section computes the
# remaining M(q)*qddot + C(q,qdot)*qdot term from the per-link mass/COM/inertia
# in ~/working_dir/kinovaDrivers/kinova-ros/kinova_description/urdf/kinova_inertial.xacro
# (links: shoulder, arm, forearm, wrist_spherical_1, wrist_spherical_2,
# hand_3finger — same 6 links, same order, as deploy_viz.py's _LINK_MESHES,
# which renders meshes using this identical FK chain against the real arm).
# Everything is done in the base/world frame (all vectors expressed in the
# same frame throughout), which is convenient since compute_jacobian already
# gives us per-joint axis + origin in that frame.
#
# Validated below (see the __main__ self-test): the resulting mass matrix
# M(q), built column-by-column via unit qddot with qdot=0, comes out symmetric
# and positive-definite (a necessary property of any valid rigid-body inertia
# matrix), and tau is exactly zero at qdot=qddot=0 for any q.
# ════════════════════════════════════════════════════════════════════════════

def _inertia_cylinder(mass, height, radius, axis=0):
    """Solid-cylinder inertia tensor about its own COM, in its own local frame.
    axis=0/1/2 selects which local axis (z/y/x) the cylinder's length runs
    along — matches the three branches of kinova_inertial.xacro's
    inertia_cylinder macro exactly."""
    i_perp = (1.0 / 12.0) * mass * (3 * radius**2 + height**2)
    i_spin = 0.5 * mass * radius**2
    if axis == 0:
        return np.diag([i_perp, i_perp, i_spin])
    if axis == 1:
        return np.diag([i_perp, i_spin, i_perp])
    return np.diag([i_spin, i_perp, i_perp])


# (mass, com_local_xyz, inertia_local) for links 1..6 = shoulder, arm, forearm,
# wrist_spherical_1, wrist_spherical_2, hand_3finger — verbatim from
# kinova_inertial.xacro. NOTE: the hand_3finger link's <mass> tag (0.99) and
# the mass plugged into its inertia formula (0.727) disagree in the upstream
# xacro itself — kept as-is rather than "corrected" without better data.
_LINK_DYNAMICS = [
    (0.7477,   np.array([0,       -0.002,     -0.0605     ]), _inertia_cylinder(0.7477, 0.14, 0.04, axis=0)),
    (0.99,     np.array([0,       -0.2065,    -0.01       ]), _inertia_cylinder(0.99,   0.35, 0.04, axis=1)),
    (0.6763,   np.array([0,        0.081,     -0.0086      ]), _inertia_cylinder(0.6763, 0.15, 0.03, axis=1)),
    (0.463,    np.array([0,        0.0028848942, -0.0541932613]), _inertia_cylinder(0.463, 0.1, 0.02, axis=0)),
    (0.463,    np.array([0,        0.0497208855, -0.0028562765]), _inertia_cylinder(0.463, 0.1, 0.02, axis=1)),
    (0.99,     np.array([0,        0,          -0.06       ]), _inertia_cylinder(0.727, 0.03, 0.04, axis=0)),
]


def fk_link_frames(q_deg: np.ndarray):
    """Like fk_frames, but also returns the full post-rotation transform of
    each link (needed for dynamics — pre_frames alone only gives the joint
    axis/origin, not the link's own orientation for its inertia tensor)."""
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    pre_frames, link_frames = [], [T.copy()]
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        T_pre = T @ _make_fk_T(xyz, rpy)
        pre_frames.append(T_pre)
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T_pre @ Tj
        link_frames.append(T.copy())
    return pre_frames, link_frames


def rnea_no_gravity(q_deg: np.ndarray, qdot_deg: np.ndarray,
                    qddot_deg: np.ndarray) -> np.ndarray:
    """Recursive Newton-Euler, gravity term omitted (Kinova's gravity-free
    torque already removes it separately). Returns the 6 joint torques
    needed to produce the given qdot/qddot from pure rigid-body dynamics —
    i.e. M(q)*qddot + C(q,qdot)*qdot, with zero assumed external force."""
    q, qdot, qddot = np.deg2rad(q_deg), np.deg2rad(qdot_deg), np.deg2rad(qddot_deg)
    pre_frames, link_frames = fk_link_frames(q_deg)

    p = np.zeros(3)              # joint/link origin, base frame
    omega = np.zeros(3)          # link angular velocity
    alpha = np.zeros(3)          # link angular acceleration
    v = np.zeros(3)              # velocity of the link origin point
    a = np.zeros(3)              # acceleration of the link origin point

    coms, oms, als, accs, Is = [], [], [], [], []
    for i in range(6):
        z_i = pre_frames[i][:3, 2]
        p_i = pre_frames[i][:3, 3]
        R_i = link_frames[i + 1][:3, :3]
        dp  = p_i - p

        omega_i = omega + qdot[i] * z_i
        alpha_i = alpha + qddot[i] * z_i + qdot[i] * np.cross(omega, z_i)
        v_i = v + np.cross(omega, dp)
        a_i = a + np.cross(alpha, dp) + np.cross(omega, np.cross(omega, dp))

        mass_i, com_local_i, I_local_i = _LINK_DYNAMICS[i]
        d_i = R_i @ com_local_i                      # COM offset from p_i, base frame
        a_com_i = a_i + np.cross(alpha_i, d_i) + np.cross(omega_i, np.cross(omega_i, d_i))
        I_world_i = R_i @ I_local_i @ R_i.T

        coms.append(p_i + d_i); oms.append(omega_i); als.append(alpha_i)
        accs.append(a_com_i);   Is.append(I_world_i)

        p, omega, alpha, v, a = p_i, omega_i, alpha_i, v_i, a_i

    # Inward pass: force/moment propagation, tip (link 6) -> base (link 1)
    tau = np.zeros(6)
    f_next, n_next, p_next = np.zeros(3), np.zeros(3), None
    joint_origins = [pf[:3, 3] for pf in pre_frames]
    joint_axes    = [pf[:3, 2] for pf in pre_frames]

    for i in range(5, -1, -1):
        mass_i = _LINK_DYNAMICS[i][0]
        F_i = mass_i * accs[i]
        N_i = Is[i] @ als[i] + np.cross(oms[i], Is[i] @ oms[i])

        f_i = F_i + f_next
        n_i = (N_i + n_next
              + np.cross(coms[i] - joint_origins[i], F_i)
              + (np.cross(p_next - joint_origins[i], f_next) if p_next is not None else 0.0))

        tau[i] = np.dot(n_i, joint_axes[i])
        f_next, n_next, p_next = f_i, n_i, joint_origins[i]

    return tau


# ════════════════════════════════════════════════════════════════════════════
# Reparametrized (first-moment / origin-inertia) RNEA — the form that is
# EXACTLY LINEAR in per-link dynamic parameters, needed to build a full
# mass+Coriolis regressor the same way gravity_regressor already does for G(q).
#
# rnea_no_gravity above parametrizes each link by (mass, com, inertia-about-COM)
# taken SEPARATELY — that makes the dynamics BILINEAR (mass multiplies a term
# that already depends on com), not linear, so it cannot be turned into a
# regressor by simply overriding those three things independently.
#
# The standard fix (Khalil & Dombre; Featherstone's "spatial inertia"; the same
# trick gravity_regressor already uses, just extended): reparametrize each
# link by the "barycentric" triple
#     mass_i          (1 param)
#     h_i = mass_i * com_i        (3 params, "first moment" -- SAME quantity as
#                                  gravity_regressor's (m*cx, m*cy, m*cz) columns)
#     J_i = inertia about the link's OWN REFERENCE-FRAME ORIGIN, not about COM
#                                  (6 independent params, symmetric tensor)
# Because a_i/omega_i/alpha_i (the kinematic quantities) never depend on this
# link's own dynamic parameters (only on upstream motion + pure geometry), the
# force/moment equations become linear and DECOUPLED in (mass_i, h_i, J_i):
#     f_i = mass_i*a_i + alpha_i x h_i + omega_i x (omega_i x h_i)
#     n_i = J_i.alpha_i + omega_i x (J_i.omega_i) + h_i x a_i
# (moments taken about the link's reference-frame origin directly, not COM --
# this is why h_i x a_i appears here but not in rnea_no_gravity's COM-based n_i).
# ════════════════════════════════════════════════════════════════════════════

def _parallel_axis_to_origin(mass, com_local, I_com_local):
    """Shifts an inertia tensor from "about COM" (how _LINK_DYNAMICS stores it,
    and how _inertia_cylinder computes it) to "about the link's own reference
    origin" (what the h-based regressor form needs) via the tensor parallel-axis
    theorem: J_origin = I_com + mass*(|r|^2*Id - r⊗r), r = vector from origin to COM."""
    r = com_local
    return I_com_local + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def _nominal_link_params_origin():
    """Converts _LINK_DYNAMICS (mass, com, inertia-about-COM) to the barycentric
    (mass, h=mass*com, inertia-about-origin) triple used by rnea_full. Computed
    once from the same nominal numbers rnea_no_gravity uses, so rnea_full at
    these nominal params must reproduce rnea_no_gravity exactly (see __main__)."""
    params = []
    for mass_i, com_i, I_com_i in _LINK_DYNAMICS:
        h_i = mass_i * com_i
        J_origin_i = _parallel_axis_to_origin(mass_i, com_i, I_com_i)
        params.append((mass_i, h_i, J_origin_i))
    return params


_NOMINAL_LINK_PARAMS_ORIGIN = _nominal_link_params_origin()


def _rnea_full_kinematics(q_deg: np.ndarray, qdot_deg: np.ndarray, qddot_deg: np.ndarray):
    """The part of rnea_full that does NOT depend on link_params -- pure
    kinematics from (q,qdot,qddot). Split out so full_dynamics_regressor can
    compute this ONCE per sample instead of 72 times (one per unit-perturbation
    call), since forward kinematics dominates rnea_full's cost otherwise."""
    qdot, qddot = np.deg2rad(qdot_deg), np.deg2rad(qddot_deg)
    pre_frames, link_frames = fk_link_frames(q_deg)

    p = np.zeros(3)
    omega = np.zeros(3)
    alpha = np.zeros(3)
    a = np.zeros(3)

    oms, als, accs_origin = [], [], []
    for i in range(6):
        z_i = pre_frames[i][:3, 2]
        p_i = pre_frames[i][:3, 3]
        dp = p_i - p

        omega_i = omega + qdot[i] * z_i
        alpha_i = alpha + qddot[i] * z_i + qdot[i] * np.cross(omega, z_i)
        a_i = a + np.cross(alpha, dp) + np.cross(omega, np.cross(omega, dp))

        oms.append(omega_i); als.append(alpha_i); accs_origin.append(a_i)
        p, omega, alpha, a = p_i, omega_i, alpha_i, a_i

    joint_origins = [pf[:3, 3] for pf in pre_frames]
    joint_axes    = [pf[:3, 2] for pf in pre_frames]
    link_rots     = [lf[:3, :3] for lf in link_frames[1:]]
    return oms, als, accs_origin, joint_origins, joint_axes, link_rots


def _rnea_full_dynamics(kin, link_params) -> np.ndarray:
    """The link_params-dependent inward pass, given precomputed kinematics
    from _rnea_full_kinematics. This is the part that's actually linear in
    link_params -- called once per unit-perturbation column by
    full_dynamics_regressor, reusing the SAME kinematics each time."""
    oms, als, accs_origin, joint_origins, joint_axes, link_rots = kin
    tau = np.zeros(6)
    f_next, n_next, p_next = np.zeros(3), np.zeros(3), None

    for i in range(5, -1, -1):
        mass_i, h_local_i, J_local_i = link_params[i]
        R_i = link_rots[i]
        h_i = R_i @ h_local_i                # first moment, base frame
        J_i = R_i @ J_local_i @ R_i.T        # inertia about link origin, base frame

        F_i = mass_i * accs_origin[i] + np.cross(als[i], h_i) + np.cross(oms[i], np.cross(oms[i], h_i))
        N_i = J_i @ als[i] + np.cross(oms[i], J_i @ oms[i]) + np.cross(h_i, accs_origin[i])

        f_i = F_i + f_next
        n_i = (N_i + n_next
              + (np.cross(p_next - joint_origins[i], f_next) if p_next is not None else 0.0))

        tau[i] = np.dot(n_i, joint_axes[i])
        f_next, n_next, p_next = f_i, n_i, joint_origins[i]

    return tau


def rnea_full(q_deg: np.ndarray, qdot_deg: np.ndarray, qddot_deg: np.ndarray,
             link_params=None) -> np.ndarray:
    """Recursive Newton-Euler (gravity excluded, same convention as
    rnea_no_gravity), but built from the (mass, h, J_origin) barycentric
    parametrization so it is EXACTLY LINEAR in those parameters -- the
    property full_dynamics_regressor exploits to build Y_full by unit
    perturbation. link_params defaults to _NOMINAL_LINK_PARAMS_ORIGIN (the
    same physical arm rnea_no_gravity describes); pass overrides to evaluate
    an arbitrary parameter set (e.g. one column's unit basis vector, or a
    fitted correction)."""
    if link_params is None:
        link_params = _NOMINAL_LINK_PARAMS_ORIGIN
    kin = _rnea_full_kinematics(q_deg, qdot_deg, qddot_deg)
    return _rnea_full_dynamics(kin, link_params)


def full_dynamics_regressor(q_deg: np.ndarray, qdot_deg: np.ndarray, qddot_deg: np.ndarray,
                            include_friction: bool = True) -> np.ndarray:
    """Y_full(q,qdot,qddot), shape (6, 10*n_links [+ 2*n_links if friction]),
    such that  M(q)*qddot + C(q,qdot)*qdot [+ friction] = Y_full @ pi
    with pi packing, per link, [mass, hx, hy, hz, Jxx, Jyy, Jzz, Jxy, Jxz, Jyz]
    (J about the link's own reference origin -- see rnea_full), plus, if
    include_friction, [viscous_j, coulomb_j] per JOINT appended at the end.

    Built by unit-perturbation of rnea_full: since rnea_full is exactly linear
    in each (mass_i, h_i, J_i) entry (by construction -- see the block comment
    above), evaluating rnea_full with ONE parameter set to 1 and all others to
    0 gives EXACTLY that parameter's column (not a finite-difference
    approximation -- linear functions have exact, step-size-independent
    "derivatives" this way). This mirrors how gravity_regressor's 4 columns
    per link are the (mass, hx, hy, hz) subset of these same first 4 columns
    (verified in __main__: full_dynamics_regressor's first 4 columns per link,
    evaluated at qdot=qddot=0 with the gravity-equivalent base acceleration,
    must reduce to gravity_regressor exactly)."""
    n_links = len(_LINK_DYNAMICS)
    n_inertial = 10 * n_links
    n_cols = n_inertial + (2 * n_links if include_friction else 0)
    Y = np.zeros((6, n_cols))

    # Kinematics (pre_frames/link_frames/omega/alpha/accel chain) depend only
    # on (q,qdot,qddot), never on link_params -- computed ONCE and reused
    # across all 72 unit-perturbation calls below, instead of recomputing
    # forward kinematics from scratch 72 times for the same sample (this was
    # the dominant cost building datasets with tens of thousands of samples).
    kin = _rnea_full_kinematics(q_deg, qdot_deg, qddot_deg)

    zero_params = [(0.0, np.zeros(3), np.zeros((3, 3))) for _ in range(n_links)]

    for i in range(n_links):
        base = 10 * i
        # mass column
        params = [p for p in zero_params]
        params[i] = (1.0, np.zeros(3), np.zeros((3, 3)))
        Y[:, base] = _rnea_full_dynamics(kin, params)
        # h columns (3)
        for k in range(3):
            h = np.zeros(3); h[k] = 1.0
            params = [p for p in zero_params]
            params[i] = (0.0, h, np.zeros((3, 3)))
            Y[:, base + 1 + k] = _rnea_full_dynamics(kin, params)
        # J columns (6 independent entries: xx,yy,zz,xy,xz,yz)
        j_slots = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for k, (r, c) in enumerate(j_slots):
            J = np.zeros((3, 3))
            J[r, c] = 1.0
            J[c, r] = 1.0
            params = [p for p in zero_params]
            params[i] = (0.0, np.zeros(3), J)
            Y[:, base + 4 + k] = _rnea_full_dynamics(kin, params)

    if include_friction:
        qdot = np.deg2rad(qdot_deg)
        for i in range(n_links):
            col = n_inertial + 2 * i
            Y[i, col] = qdot[i]                    # viscous: Fv_i * qdot_i
            Y[i, col + 1] = np.sign(qdot[i]) if abs(qdot[i]) > 1e-6 else 0.0  # Coulomb: Fc_i * sign(qdot_i)

    return Y


# ════════════════════════════════════════════════════════════════════════════
# Gravity-residual model (linear-in-parameters) for contact-force RECOVERY
#
# GetAngularForceGravityFree strips gravity g(q) using the factory link
# mass/COM parameters, but those don't perfectly match the physical arm, so a
# smooth, pose-dependent residual (~1-2.5 N mapped to the EEF) survives even in
# free space with no contact. A separate ~0.1-0.5 N pose-agnostic jitter is the
# true sensor noise floor. The mounted tool (~6 g) is negligible and is NOT the
# residual source — this models ARM-LINK gravity-model error, so recalibrate on
# arm reconfiguration, not on a tool swap of that scale.
#
# That residual is itself a gravity torque, hence LINEAR in a set of per-link
# inertial-parameter corrections:
#
#     tau_residual(q) = Y_g(q) @ phi
#
# where Y_g(q) (the "gravity regressor") is built analytically from kinematics
# (NOT learned) and phi packs, per link, (m, m*cx, m*cy, m*cz) best-fit
# corrections. phi is fit from data in calibrate_gravity_residual.py; it is NOT
# the arm's true inertial parameters (static gravity data can't identify all of
# them — the fit is ridge-regularized), just a correction that best explains
# the leftover torque.
#
# Full model, in subtraction order:
#     tau_gf = Y_g(q) @ phi  +  [M(q)qddot + C(q,qdot)qdot]  +  tau_ext
# The middle term is rnea_no_gravity (near-zero in the quasi-static regime).
# tau_ext is the external contact torque we want; map to an EEF wrench with the
# existing torque_to_wrench (Jacobian-transpose pinv).
#
# GRAVITY AXIS: for the Jaco2 mounted upright, base +Z is "up", so gravity
# points along -Z. The __main__ self-test only checks INTERNAL consistency
# (Y_g @ phi == independent gravity_rnea), which holds for ANY choice of
# G_BASE — the physically-correct axis/sign is confirmed EMPIRICALLY (probe:
# raw - gravity_free torque ≈ gravity_rnea(nominal); and calibrate's
# before/after residual drop). Do not treat the self-test as axis validation.
# ════════════════════════════════════════════════════════════════════════════

G_BASE = np.array([0.0, 0.0, -9.81])   # base-frame gravity (m/s^2), +Z up


def _skew(v):
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0],  0.0]])


def gravity_rnea(q_deg: np.ndarray, masses=None, coms=None) -> np.ndarray:
    """Independent RNEA-with-gravity via the base-acceleration trick: returns
    the 6 joint torques G(q) needed to hold the arm against gravity at
    qdot=qddot=0. Implemented as an iterative force/moment propagation —
    structurally distinct from gravity_regressor's closed-form moment sum — so
    the __main__ self-test that cross-checks the two actually catches bugs.
    masses/coms default to _LINK_DYNAMICS; pass overrides to evaluate an
    arbitrary phi. Degrees in, converted internally (matches fk_link_frames)."""
    if masses is None:
        masses = [ld[0] for ld in _LINK_DYNAMICS]
    if coms is None:
        coms = [ld[1] for ld in _LINK_DYNAMICS]
    pre_frames, link_frames = fk_link_frames(q_deg)
    joint_origins = [pf[:3, 3] for pf in pre_frames]
    joint_axes    = [pf[:3, 2] for pf in pre_frames]

    a0 = -G_BASE                     # fictitious base accel representing gravity
    coms_pos, Fs = [], []
    for i in range(6):
        R_i = link_frames[i + 1][:3, :3]
        p_i = joint_origins[i]
        coms_pos.append(p_i + R_i @ coms[i])
        Fs.append(masses[i] * a0)

    tau = np.zeros(6)
    f_next, n_next, p_next = np.zeros(3), np.zeros(3), None
    for i in range(5, -1, -1):
        F_i = Fs[i]
        f_i = F_i + f_next
        n_i = (np.cross(coms_pos[i] - joint_origins[i], F_i) + n_next
               + (np.cross(p_next - joint_origins[i], f_next) if p_next is not None else 0.0))
        tau[i] = np.dot(n_i, joint_axes[i])
        f_next, n_next, p_next = f_i, n_i, joint_origins[i]
    return tau


def gravity_regressor(q_deg: np.ndarray) -> np.ndarray:
    """Gravity regressor Y_g(q), shape (6, 4*n_links), so tau_gravity = Y_g @ phi
    with phi packing per link [m, m*cx, m*cy, m*cz]. Closed form: joint j's
    gravity torque is the sum, over links i OUTBOARD of j (i >= j), of the
    moment of link i's gravitational force about joint j's axis — linear in
    each link's (m, m*c), giving 4 columns per link. Built analytically from
    the same fk_link_frames used everywhere else (degrees in, converted
    internally, so it composes consistently with FK/Jacobian/RNEA)."""
    n_links = len(_LINK_DYNAMICS)
    pre_frames, link_frames = fk_link_frames(q_deg)
    joint_origins = [pf[:3, 3] for pf in pre_frames]
    joint_axes    = [pf[:3, 2] for pf in pre_frames]
    a0 = -G_BASE
    Sx = _skew(a0)

    Y = np.zeros((6, 4 * n_links))
    for i in range(n_links):
        R_i = link_frames[i + 1][:3, :3]
        p_i = joint_origins[i]
        for j in range(i + 1):        # joint j <= link i (link i is outboard of joint j)
            z_j = joint_axes[j]
            o_j = joint_origins[j]
            # mass column: d(tau_j)/d(m_i) = z_j . ((p_i - o_j) x a0)
            Y[j, 4 * i] = np.dot(z_j, np.cross(p_i - o_j, a0))
            # m*c columns: term = z_j . ((R_i w) x a0) = -z_j^T skew(a0) R_i w
            Y[j, 4 * i + 1:4 * i + 4] = -z_j @ Sx @ R_i
    return Y


def recover_external_force(q_deg: np.ndarray, tau_gf: np.ndarray, phi: np.ndarray,
                           qdot_deg: np.ndarray = None, qddot_deg: np.ndarray = None,
                           compensate_dynamics: bool = False,
                           damping: float = 0.0):
    """Clean-force recovery entry point (Deliverable 4). Given one (q, tau_gf)
    reading and a fitted gravity-regressor phi (from calibrate_gravity_residual.py),
    returns (tau_ext, F_ext):

        tau_ext = tau_gf - Y_g(q)@phi  [ - rnea_no_gravity(q,qdot,qddot) if compensate_dynamics ]
        F_ext   = pinv(J(q)^T) @ tau_ext   (or damped, if damping>0 — see torque_to_wrench)

    tau_ext: (6,) clean external joint torque, N*m.
    F_ext:   (6,) [Fx,Fy,Fz,Mx,My,Mz] clean external EEF wrench, base frame.
             Force-magnitude convention matches the rest of this module:
             use only F_ext[:3] (linear part) for ||F||.

    This is a stateless function (unlike ContactDetector) — suitable for
    replaying a logged stream of (q, tau_gf) pairs, e.g. from
    replay_episode.py's torque_log.npz, not just live use."""
    tau_ext = tau_gf.astype(np.float64) - gravity_regressor(q_deg) @ phi
    if compensate_dynamics:
        qdot_deg  = qdot_deg  if qdot_deg  is not None else np.zeros(6)
        qddot_deg = qddot_deg if qddot_deg is not None else np.zeros(6)
        tau_ext = tau_ext - rnea_no_gravity(q_deg, qdot_deg, qddot_deg)
    F_ext = torque_to_wrench(q_deg, tau_ext, damping=damping)
    return tau_ext, F_ext


class VelocityDifferentiator:
    """Estimates joint acceleration (deg/s^2) from consecutive angular
    velocity readings (deg/s), since the Kinova SDK doesn't expose qddot
    directly (GetActuatorAcceleration is a per-joint 3-axis accelerometer in
    G's, not joint angular acceleration). A single differentiation of the
    SDK-provided velocity is far less noisy than double-differentiating
    position, but is still a finite difference — smoothed with a short
    exponential moving average to tame sensor jitter."""
    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing   # 0 = no smoothing, closer to 1 = heavier
        self._last_qdot = None
        self._last_t = None
        self._qddot_smooth = np.zeros(6)

    def update(self, qdot_deg: np.ndarray, t: float) -> np.ndarray:
        if self._last_qdot is None or t <= self._last_t:
            self._last_qdot, self._last_t = qdot_deg.copy(), t
            return self._qddot_smooth
        raw = (qdot_deg - self._last_qdot) / (t - self._last_t)
        self._qddot_smooth = (self.smoothing * self._qddot_smooth
                              + (1 - self.smoothing) * raw)
        self._last_qdot, self._last_t = qdot_deg.copy(), t
        return self._qddot_smooth


class MovingBaseline:
    """
    Continuously-tracked exponential moving average (EMA) of the per-joint
    gravity-free torque, used as an ADAPTIVE alternative to a single fixed
    calibrated bias. A fixed bias only cancels the residual at the ONE pose it
    was captured at (see ContactDetector.calibrate) — moving to a different
    pose reintroduces pose-dependent residual error. A slow-tracking average
    instead follows that drift continuously, so (signal - running_average)
    stays near zero through ordinary pose changes and only spikes on
    something genuinely sudden, like real contact.

    time_constant: seconds. Must be much slower than a real contact
    transient (which develops over a handful of control ticks) but fast
    enough to track genuine pose-change drift (which develops over ~1s+ of
    arm motion) — default 2.0s sits between those.

    IMPORTANT TRADEOFF (confirmed empirically, not just in theory): for a
    torque ramping at `rate` (N·m/s) due to pose change, the EMA lags it in
    steady state by exactly `rate * time_constant`. If that lag exceeds
    enter_threshold, the drift itself gets misread as contact — a synthetic
    test with a 2 N·m/s ramp and time_constant=2.0s produced a false ENTER
    partway through the ramp (lag settles at 4 N·m > a 3 N·m threshold);
    dropping to time_constant=1.0s (lag=2 N·m) or 0.5s (lag=1 N·m) fixed it
    and left a clean, correctly-timed ENTER/EXIT around the real spike only.
    Rule of thumb: pick time_constant well under enter_threshold / (expected
    pose-drift rate in N·m/s) for your setup.

    IMPORTANT: freeze updates while in_contact (see ContactDetector.update),
    otherwise a sustained press would eventually get absorbed into the
    average and the detector would "forget" it's still in contact.
    """
    def __init__(self, dim: int = 6, time_constant: float = 2.0):
        self.time_constant = time_constant
        self.value = np.zeros(dim)
        self._initialized = False
        self._last_t = None

    def seed(self, value: np.ndarray):
        self.value = np.asarray(value, dtype=np.float64).copy()
        self._initialized = True

    def update(self, x: np.ndarray, t: float, freeze: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self._initialized:
            self.value = x.copy()
            self._initialized = True
            self._last_t = t
            return self.value
        if self._last_t is None:
            # seed() set a value but we've never had a timestamp yet (e.g.
            # calibrate() ran before the first real update()) — just record
            # it now rather than computing a bogus dt against None.
            self._last_t = t
            return self.value
        dt = max(t - self._last_t, 1e-6)
        self._last_t = t
        if not freeze:
            alpha = np.exp(-dt / self.time_constant)   # dt-aware, poll-rate independent
            self.value = alpha * self.value + (1 - alpha) * x
        return self.value


# ════════════════════════════════════════════════════════════════════════════
# ContactDetector
# ════════════════════════════════════════════════════════════════════════════

class ContactDetector:
    """
    mode='force': trigger on ||[Fx,Fy,Fz]|| (Jacobian-mapped EEF contact force, N)
    mode='joint': trigger on max(|tau_i|) across joints (N·m) — use this if you
                  don't trust the Jacobian near a singular arm configuration.

    baseline_mode='fixed' (default, unchanged from before): subtract a single
        bias vector captured once via calibrate()/set_bias(). Simple, and
        fine if the arm stays near the calibration pose — but pose-dependent
        residual creeps back in away from it (see MovingBaseline docstring).
    baseline_mode='moving': subtract a continuously-updated MovingBaseline
        instead, so pose-related drift is tracked away automatically and only
        sudden spikes (real contact) register. Requires a timestamp `t` each
        update(). Both modes share the same calibrate()/set_bias() call —
        calibrate() seeds the moving average too, so switching modes doesn't
        need a different calibration flow.
    baseline_mode='gravity_model': subtract Y_g(q) @ phi (the fitted,
        ridge-regularized gravity-residual model — see calibrate_gravity_residual.py
        and gravity_regressor above) evaluated FRESH at the current pose every
        call — not a snapshot like 'fixed', not an EMA like 'moving'. This is
        the correct mode for continuous FORCE RECOVERY (not just binary contact
        detection): it doesn't lag a ramp and doesn't drift with pose. Requires
        set_phi()/load_phi() before use.

        DO NOT use baseline_mode='moving' for force recovery — the EMA absorbs
        any signal slower than its time_constant, which includes slow
        quasi-static contact force during replay. It would silently eat the
        very signal you're trying to recover. 'moving' remains valid ONLY for
        binary contact detection (see its own docstring), never for recovery.

    compensate_dynamics: if True, also subtract the rigid-body motion torque
        M(q)*qddot + C(q,qdot)*qdot (via rnea_no_gravity) before thresholding —
        cancels the "still see force while just moving, no contact" effect.
        Composable with any baseline_mode. Requires qdot_deg (and qddot_deg,
        or a VelocityDifferentiator to estimate it) each update().

    enter_threshold: magnitude that flips clear -> contact
    exit_ratio:      exit_threshold = enter_threshold * exit_ratio (< 1, hysteresis)
    debounce:        consecutive samples required before flipping state either way
    """
    def __init__(self, enter_threshold: float = 5.0, exit_ratio: float = 0.6,
                 debounce: int = 3, mode: str = 'force',
                 compensate_dynamics: bool = False,
                 baseline_mode: str = 'fixed', moving_time_constant: float = 2.0):
        assert mode in ('force', 'joint')
        assert baseline_mode in ('fixed', 'moving', 'gravity_model')
        self.mode = mode
        self.compensate_dynamics = compensate_dynamics
        self.baseline_mode = baseline_mode
        self.enter_threshold = enter_threshold
        self.exit_threshold  = enter_threshold * exit_ratio
        self.debounce = debounce

        self.bias = None            # (6,) fixed residual torque bias from calibrate()
        self.phi  = None            # (4*n_links,) gravity-regressor params from calibrate_gravity_residual.py
        self._moving = MovingBaseline(dim=6, time_constant=moving_time_constant) \
            if baseline_mode == 'moving' else None
        self.in_contact = False
        self._streak = 0

    def set_bias(self, bias: np.ndarray):
        """Set a previously-calibrated bias directly (see calibrate_contact_baseline.py) —
        use this to reuse a saved calibration instead of recapturing it every run.
        Also seeds the moving baseline (if baseline_mode='moving') to the same value."""
        self.bias = np.asarray(bias, dtype=np.float64)
        if self._moving is not None:
            self._moving.seed(self.bias)

    def set_phi(self, phi: np.ndarray):
        """Set the fitted gravity-regressor parameter vector directly (for
        baseline_mode='gravity_model')."""
        self.phi = np.asarray(phi, dtype=np.float64)

    def load_phi(self, path: str):
        """Load phi saved by calibrate_gravity_residual.py."""
        self.set_phi(np.load(path))

    def calibrate(self, gf_torque_samples: np.ndarray):
        """gf_torque_samples: (N, 6) gravity-free torque readings collected
        while the arm is stationary and NOT touching anything. Removes the
        residual bias left over from imperfect gravity compensation (and,
        with compensate_dynamics, from imperfect link mass/inertia values —
        both are static-pose systematic errors, so a single-pose bias capture
        removes them the same way). NOTE: this bias is tool-specific (it
        includes whatever mass the gripper is currently holding) and
        pose-specific (see contact_detector module docstring) — recalibrate
        whenever the attached tool changes. With baseline_mode='moving' this
        is just the STARTING point; the running average takes over from here."""
        self.set_bias(np.asarray(gf_torque_samples).mean(axis=0))

    def magnitude(self, tau_gf: np.ndarray, q_deg: np.ndarray,
                 qdot_deg: np.ndarray = None, qddot_deg: np.ndarray = None):
        """Returns (magnitude, wrench_or_None, tau_residual). Does NOT advance
        the moving baseline (see update(), which does, in the right order)."""
        tau = tau_gf.astype(np.float64)
        if self.compensate_dynamics:
            qdot_deg  = qdot_deg  if qdot_deg  is not None else np.zeros(6)
            qddot_deg = qddot_deg if qddot_deg is not None else np.zeros(6)
            tau = tau - rnea_no_gravity(q_deg, qdot_deg, qddot_deg)

        if self.baseline_mode == 'moving':
            reference = self._moving.value if self._moving._initialized else tau
        elif self.baseline_mode == 'gravity_model':
            assert self.phi is not None, \
                "baseline_mode='gravity_model' requires set_phi()/load_phi() first"
            reference = gravity_regressor(q_deg) @ self.phi
        else:
            reference = self.bias if self.bias is not None else 0.0
        tau = tau - reference

        if self.mode == 'joint':
            return float(np.max(np.abs(tau))), None, tau
        F = torque_to_wrench(q_deg, tau)
        return float(np.linalg.norm(F[:3])), F, tau

    def update(self, tau_gf: np.ndarray, q_deg: np.ndarray,
              qdot_deg: np.ndarray = None, qddot_deg: np.ndarray = None,
              t: float = None):
        """Feed one new reading. Returns (in_contact, magnitude, wrench_or_None,
        changed) — `changed` is True on the sample where the state flips, so
        callers can log/print exactly once per transition.

        t (seconds, e.g. time.time()) is REQUIRED when baseline_mode='moving'
        — used to advance the running average by the right amount of wall-clock
        time regardless of poll rate, and frozen (not advanced) while in
        contact so a sustained press isn't absorbed into the baseline."""
        mag, F, _tau_residual = self.magnitude(tau_gf, q_deg, qdot_deg, qddot_deg)
        threshold = self.exit_threshold if self.in_contact else self.enter_threshold
        crossed = (mag < threshold) if self.in_contact else (mag > threshold)

        changed = False
        if crossed:
            self._streak += 1
            if self._streak >= self.debounce:
                self.in_contact = not self.in_contact
                self._streak = 0
                changed = True
        else:
            self._streak = 0

        if self.baseline_mode == 'moving':
            assert t is not None, "baseline_mode='moving' requires t=<timestamp> each update()"
            self._moving.update(tau_gf.astype(np.float64), t, freeze=self.in_contact)

        return self.in_contact, mag, F, changed


# ════════════════════════════════════════════════════════════════════════════
# Self-tests — run: python contact_detector.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    ok = True

    # 1. RNEA: zero torque at rest (qdot=qddot=0), for any pose.
    max_rest = max(np.max(np.abs(rnea_no_gravity(rng.uniform(-150, 150, 6),
                                                 np.zeros(6), np.zeros(6))))
                   for _ in range(20))
    print(f'[rnea]    max |tau| at rest, 20 poses: {max_rest:.2e}   (expect ~0)')
    ok &= max_rest < 1e-9

    # 2. RNEA: mass matrix symmetric + positive-definite (real inertia property).
    for _ in range(3):
        q = rng.uniform(-150, 150, 6)
        M = np.column_stack([rnea_no_gravity(q, np.zeros(6), np.eye(6)[k]) for k in range(6)])
        sym = np.max(np.abs(M - M.T))
        min_eig = np.linalg.eigvalsh(0.5 * (M + M.T)).min()
        print(f'[rnea]    M(q) symmetry err={sym:.2e}  min eig={min_eig:.3e}   (expect ~0, >0)')
        ok &= (sym < 1e-9 and min_eig > 0)

    # 3. Gravity regressor: Y_g(q) @ phi_nominal == independent gravity_rnea(q).
    #    Cross-checks the closed-form regressor against the iterative RNEA — a
    #    real test since they're implemented differently. (Confirms the
    #    factorization/kinematics, NOT the physical gravity axis; see G_BASE.)
    phi_nom = np.concatenate([[m, *(m * c)] for (m, c, _I) in _LINK_DYNAMICS])
    max_err = max(np.max(np.abs(gravity_regressor(q := rng.uniform(-180, 180, 6)) @ phi_nom
                                - gravity_rnea(q)))
                  for _ in range(50))
    print(f'[gravity] max |Y_g@phi_nom - gravity_rnea|, 50 poses: {max_err:.2e}   (expect <1e-6)')
    ok &= max_err < 1e-6

    # 4. rnea_full (barycentric/origin-inertia form) at nominal params must
    #    reproduce rnea_no_gravity (COM/mass form) EXACTLY -- same physics,
    #    different parametrization. This is THE critical check that the
    #    reparametrization (needed to make the regressor linear) didn't
    #    silently change the dynamics.
    max_reparam_err = 0.0
    for _ in range(30):
        q, qdot, qddot = rng.uniform(-150, 150, 6), rng.uniform(-60, 60, 6), rng.uniform(-200, 200, 6)
        diff = np.max(np.abs(rnea_full(q, qdot, qddot) - rnea_no_gravity(q, qdot, qddot)))
        max_reparam_err = max(max_reparam_err, diff)
    print(f'[dynamics] max |rnea_full(nominal) - rnea_no_gravity|, 30 samples: '
          f'{max_reparam_err:.2e}   (expect <1e-8)')
    ok &= max_reparam_err < 1e-8

    # 5. full_dynamics_regressor @ pi_nominal == rnea_full(nominal), i.e. the
    #    unit-perturbation column construction actually reproduces direct
    #    evaluation when contracted with the true parameter vector -- the
    #    regressor-construction analog of test 3 above.
    def _pack_pi_nominal(link_params):
        j_slots = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        pi = []
        for mass_i, h_i, J_i in link_params:
            pi += [mass_i, h_i[0], h_i[1], h_i[2]] + [J_i[r, c] for r, c in j_slots]
        pi += [0.0] * (2 * len(link_params))   # zero friction (not part of the nominal model)
        return np.array(pi)

    pi_nom = _pack_pi_nominal(_NOMINAL_LINK_PARAMS_ORIGIN)
    max_regressor_err = 0.0
    for _ in range(30):
        q, qdot, qddot = rng.uniform(-150, 150, 6), rng.uniform(-60, 60, 6), rng.uniform(-200, 200, 6)
        diff = np.max(np.abs(full_dynamics_regressor(q, qdot, qddot) @ pi_nom
                             - rnea_full(q, qdot, qddot)))
        max_regressor_err = max(max_regressor_err, diff)
    print(f'[dynamics] max |Y_full@pi_nom - rnea_full(nominal)|, 30 samples: '
          f'{max_regressor_err:.2e}   (expect <1e-8)')
    ok &= max_regressor_err < 1e-8

    # 6. At qdot=qddot=0, M(q)*0 + C(q,0)*0 must be exactly zero regardless of
    #    parameters -- a quick sanity check on the regressor (not just the
    #    nominal-contraction test above).
    max_zero_motion = max(np.max(np.abs(full_dynamics_regressor(
        rng.uniform(-150, 150, 6), np.zeros(6), np.zeros(6), include_friction=False)))
        for _ in range(10))
    print(f'[dynamics] max |Y_full| at qdot=qddot=0, 10 poses: {max_zero_motion:.2e}   (expect ~0)')
    ok &= max_zero_motion < 1e-9

    print('\nALL SELF-TESTS PASSED' if ok else '\nSELF-TEST FAILURE')
    import sys as _sys
    _sys.exit(0 if ok else 1)
