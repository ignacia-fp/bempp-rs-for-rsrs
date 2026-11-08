//! Persistent high-level Bempp-rs operators exposed to Python.

use std::{
    mem,
    sync::{
        mpsc::{self, Sender},
        OnceLock,
    },
    thread::{self, JoinHandle},
};

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions,
    function::{FunctionSpace, FunctionSpaceTrait},
    greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator,
    laplace,
};
use green_kernels::types::GreenKernelEvalType;
use mpi::{environment::Threading, topology::SimpleCommunicator};
use ndelement::{
    ciarlet::LagrangeElementFamily,
    types::{Continuity, ReferenceCellType},
};
use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rlst::prelude::*;

static MPI_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_mpi() -> Result<(), String> {
    MPI_INITIALIZED
        .get_or_init(|| {
            if let Some((universe, provided)) = mpi::initialize_with_threading(Threading::Multiple) {
                if provided < Threading::Serialized {
                    return Err(format!(
                        "MPI provides {provided:?}, but Bempp-rs Python operators require at least Serialized"
                    ));
                }
                // Python owns the process lifetime. Deliberately keep MPI alive
                // until process exit instead of finalising while native objects
                // or other extension modules may still use it.
                mem::forget(universe);
            }
            Ok(())
        })
        .clone()
}

enum Command {
    Apply {
        columns: usize,
        values: Vec<f64>,
        transpose: bool,
        response: Sender<Result<Vec<f64>, String>>,
    },
    Stop,
}

struct Ready {
    dimension: usize,
    sender: Sender<Command>,
}

fn operator_worker(
    refinement_level: u32,
    function_space_degree: usize,
    quadrature_degree: usize,
    singular_quadrature_degree: usize,
    local_tree_depth: usize,
    global_tree_depth: usize,
    expansion_order: usize,
    ready: Sender<Result<Ready, String>>,
) {
    if let Err(error) = ensure_mpi() {
        let _ = ready.send(Err(error));
        return;
    }

    let communicator = SimpleCommunicator::self_comm();
    let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &communicator);
    let continuity = if function_space_degree == 0 {
        Continuity::Discontinuous
    } else {
        Continuity::Standard
    };
    let family = LagrangeElementFamily::<f64>::new(function_space_degree, continuity);
    let space = FunctionSpace::new(&grid, &family);

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quadrature_degree);
    options.set_singular_quadrature_degree(
        (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
        singular_quadrature_degree,
    );
    let quadrature = options.get_regular_quadrature_rule(ReferenceCellType::Triangle);
    let quadrature_points = quadrature.points.iter().copied().collect::<Vec<f64>>();

    let kernel = KiFmmEvaluator::from_spaces(
        &space,
        &space,
        GreenKernelEvalType::Value,
        local_tree_depth,
        global_tree_depth,
        expansion_order,
        &quadrature_points,
    );
    let operator = laplace::evaluator::single_layer(&space, &space, kernel.r(), &options);
    let dimension = space.global_dof_count();

    let (sender, receiver) = mpsc::channel();
    if ready.send(Ok(Ready { dimension, sender })).is_err() {
        return;
    }

    while let Ok(command) = receiver.recv() {
        match command {
            Command::Stop => break,
            Command::Apply {
                columns,
                values,
                transpose,
                response,
            } => {
                let result = if values.len() != dimension * columns {
                    Err(format!(
                        "received {} values for shape [{dimension}, {columns}]",
                        values.len()
                    ))
                } else {
                    let mode = if transpose {
                        TransMode::Trans
                    } else {
                        TransMode::NoTrans
                    };
                    let mut output = vec![0.0; values.len()];
                    for column in 0..columns {
                        let start = column * dimension;
                        let end = start + dimension;
                        let mut input = zero_element(operator.domain());
                        input
                            .view_mut()
                            .local_mut()
                            .data_mut()
                            .copy_from_slice(&values[start..end]);
                        let result = operator.apply(input.r(), mode);
                        output[start..end].copy_from_slice(result.view().local().data());
                    }
                    Ok(output)
                };
                let _ = response.send(result);
            }
        }
    }
}

/// Persistent Laplace single-layer operator on a regular sphere.
#[pyclass]
struct LaplaceSingleLayer {
    dimension: usize,
    sender: Sender<Command>,
    worker: Option<JoinHandle<()>>,
}

#[pymethods]
impl LaplaceSingleLayer {
    #[new]
    #[pyo3(signature = (
        refinement_level,
        function_space_degree=0,
        quadrature_degree=6,
        singular_quadrature_degree=4,
        local_tree_depth=1,
        global_tree_depth=6,
        expansion_order=6,
    ))]
    fn new(
        refinement_level: u32,
        function_space_degree: usize,
        quadrature_degree: usize,
        singular_quadrature_degree: usize,
        local_tree_depth: usize,
        global_tree_depth: usize,
        expansion_order: usize,
    ) -> PyResult<Self> {
        if expansion_order < 2 {
            return Err(PyRuntimeError::new_err(
                "expansion_order must be at least 2",
            ));
        }
        let (ready_sender, ready_receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            operator_worker(
                refinement_level,
                function_space_degree,
                quadrature_degree,
                singular_quadrature_degree,
                local_tree_depth,
                global_tree_depth,
                expansion_order,
                ready_sender,
            )
        });
        let ready = ready_receiver
            .recv()
            .map_err(|_| PyRuntimeError::new_err("Bempp-rs operator worker stopped during setup"))?
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            dimension: ready.dimension,
            sender: ready.sender,
            worker: Some(worker),
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn matvec<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input = values.as_slice()?.to_vec();
        if input.len() != self.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "vector has length {}, expected {}",
                input.len(),
                self.dimension
            )));
        }
        let result = py.allow_threads(|| self.apply(1, input, false))?;
        Ok(result.into_pyarray_bound(py))
    }

    fn transpose_matvec<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input = values.as_slice()?.to_vec();
        if input.len() != self.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "vector has length {}, expected {}",
                input.len(),
                self.dimension
            )));
        }
        let result = py.allow_threads(|| self.apply(1, input, true))?;
        Ok(result.into_pyarray_bound(py))
    }

    fn matmat<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.apply_matrix(py, values, false)
    }

    fn transpose_matmat<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.apply_matrix(py, values, true)
    }
}

impl LaplaceSingleLayer {
    fn apply(&self, columns: usize, values: Vec<f64>, transpose: bool) -> PyResult<Vec<f64>> {
        let (response_sender, response_receiver) = mpsc::channel();
        self.sender
            .send(Command::Apply {
                columns,
                values,
                transpose,
                response: response_sender,
            })
            .map_err(|_| PyRuntimeError::new_err("Bempp-rs operator worker has stopped"))?;
        response_receiver
            .recv()
            .map_err(|_| PyRuntimeError::new_err("Bempp-rs operator worker dropped its result"))?
            .map_err(PyRuntimeError::new_err)
    }

    fn apply_matrix<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = values.as_array();
        let (rows, columns) = view.dim();
        if rows != self.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "matrix has {rows} rows, expected {}",
                self.dimension
            )));
        }
        // Transfer columns contiguously to the worker. This is independent of
        // the incoming NumPy C/Fortran memory order.
        let mut input = Vec::with_capacity(rows * columns);
        for column in 0..columns {
            input.extend(view.column(column).iter().copied());
        }
        let output = py.allow_threads(|| self.apply(columns, input, transpose))?;
        let mut result = Array2::zeros((rows, columns));
        for column in 0..columns {
            let start = column * rows;
            let end = start + rows;
            result
                .column_mut(column)
                .iter_mut()
                .zip(output[start..end].iter())
                .for_each(|(target, source)| *target = *source);
        }
        Ok(result.into_pyarray_bound(py))
    }
}

impl Drop for LaplaceSingleLayer {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[pymodule]
fn _bempp_rs_operators(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LaplaceSingleLayer>()?;
    Ok(())
}
