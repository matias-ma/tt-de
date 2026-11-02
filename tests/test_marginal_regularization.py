import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from ftde.direct_training import DirectTrainingProblem, NoisyDirectTrainingProblem
from ftde.gaussian_convolution import GaussianConvolution
from ftde.simple_initialization import init_tt_constant


def test_marginal_curvature_matches_independent_density_cubature_and_gradient():
    box = np.array([[-1., 2.], [-2., 2.]])
    model = init_tt_constant(box, rank=3, modes=4, perturbation=1.)
    scales = np.array([.2, .3])
    problem = DirectTrainingProblem(model, marginal_strength=1., length_scales=scales)
    t, w = roots_legendre(10)
    nodes = [(a+b)/2+(b-a)*t/2 for a, b in box]
    weights = [w*np.ptp(b)/2 for b in box]
    points = np.stack(np.meshgrid(*nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    density = lambda x: model(x[None])[0]**2
    z = np.sum(np.asarray(jax.vmap(density)(jnp.asarray(points))).reshape(10, 10)*np.outer(*weights))
    hess = np.asarray(jax.vmap(jax.hessian(density))(jnp.asarray(points))).reshape(10, 10, 2, 2)/z
    d0 = hess[:, :, 0, 0]@weights[1]
    d1 = weights[0]@hess[:, :, 1, 1]
    expected = sum(np.ptp(box[k])*scales[k]**4*np.dot(weights[k], v**2) for k, v in enumerate((d0, d1)))
    np.testing.assert_allclose(problem.marginal_penalty(problem.params), expected, rtol=1e-10)
    direction = jnp.asarray(np.random.default_rng(2).normal(size=problem.params[0].shape))
    def line(t):
        return problem.marginal_penalty((problem.params[0]+t*direction, problem.params[1]))
    np.testing.assert_allclose(jax.grad(line)(0.), (line(1e-5)-line(-1e-5))/2e-5, rtol=1e-7)
    # A nontrivial internal gauge and overall amplitude scaling preserve p.
    gauge = jnp.diag(jnp.array([.5, 2., 3.]))
    p = problem.params
    changed = (jnp.einsum('ain,ij->ajn', p[0], gauge)*7,
               jnp.einsum('ij,jkn->ikn', jnp.linalg.inv(gauge), p[1]))
    np.testing.assert_allclose(problem.marginal_penalty(changed), expected, rtol=1e-10)


def test_constant_and_noisy_loss_integration():
    model = init_tt_constant(np.tile([-1., 1.], (2, 1)), rank=2, modes=4, perturbation=.5)
    forward = GaussianConvolution.from_tt(model, [.2, .3], quadrature_order=30)
    base = NoisyDirectTrainingProblem(model, forward)
    penalized = NoisyDirectTrainingProblem(model, forward, marginal_strength=.01)
    x = base.features(np.zeros((4, 2)))
    np.testing.assert_allclose(penalized.loss(penalized.params, x)-base.loss(base.params, x),
                               penalized.marginal_penalty(penalized.params), atol=1e-12)
    constant = init_tt_constant(np.tile([-1., 1.], (2, 1)), rank=1, modes=2, perturbation=0.)
    p = DirectTrainingProblem(constant, marginal_strength=1.)
    assert p.marginal_penalty(p.params) == 0


def test_univariate_grid_matches_pair_integration_and_normalizes():
    from ftde.marginals import squared_pair_grid, squared_univariate_grid
    model = init_tt_constant(np.tile([-1., 1.], (3, 1)), rank=3, modes=5, perturbation=1.)
    x, w = roots_legendre(12)
    for axis in range(3):
        marginal = squared_univariate_grid(model, axis, x)
        pair = squared_pair_grid(model, 0, axis, x, x) if axis else squared_pair_grid(model, 0, 2, x, x)
        expected = pair@w if axis else w@pair
        np.testing.assert_allclose(marginal, expected, atol=1e-12)
        np.testing.assert_allclose(w@marginal, 1., atol=1e-12)
        np.testing.assert_array_equal(squared_univariate_grid(model, axis, [-1.1, 1.1]), [0., 0.])


def test_marginal_penalty_is_invariant_to_coordinate_units():
    model = init_tt_constant(np.array([[-1., 2.], [-2., 2.]]), rank=2, modes=4, perturbation=1.)
    scale = np.array([3., .2])
    newbox = np.asarray(model.domain)*scale[:, None]
    changed = type(model)(cores=tuple(type(c)(c.coeffs, tuple(box)) for c, box in zip(model.cores, newbox, strict=True)),
                          domain=jnp.asarray(newbox))
    a = DirectTrainingProblem(model, marginal_strength=.01)
    b = DirectTrainingProblem(changed, marginal_strength=.01)
    np.testing.assert_allclose(a.marginal_penalty(a.params), b.marginal_penalty(b.params), rtol=1e-11)
    base = DirectTrainingProblem(model)
    data = base.features(np.zeros((3, 2)))
    np.testing.assert_allclose(a.loss(a.params, data)-base.loss(base.params, data),
                               a.marginal_penalty(a.params), atol=1e-12)
